import os
import pickle
import numpy as np
import numpy.linalg as la
import PIL.Image
import PIL.ImageSequence
import dnnlib
import dnnlib.tflib as tflib
from IPython.display import display, clear_output
import moviepy
import moviepy.editor
import math
import glob
import csv
from functools import partial
import time
import collections

import tensorflow as tf

import keras
from keras.applications.vgg16 import VGG16, preprocess_input

from sklearn.linear_model import LinearRegression, Lasso

import colorsys
import requests
import re
import copy

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


##
# Load network snapshot
##

#input_sg_name = "2019-02-09-stylegan-danbooru2017-faces-network-snapshot-007841.pkl"

# From https://mega.nz/#!vOgj1QoD!GD3E37BroNnZaIR_nic2zVxBtKfAqlvbEC8uBK8-4co
input_sg_name = "2019-02-18-stylegan-faces-network-02041-011095.pkl"

tflib.init_tf()

# Load pre-trained network.
with open(input_sg_name, 'rb') as f:
    # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
    # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
    # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.    
    _G, _D, Gs = pickle.load(f)
        
# Print network details.
Gs.print_layers()
_D.print_layers()


##
# Build things on top for encoding
# Based on https://github.com/Puzer/stylegan
##
def create_stub(name, batch_size):
    return tf.constant(0, dtype='float32', shape=(batch_size, 0))

dlatent_avg = tf.get_default_session().run(Gs.own_vars["dlatent_avg"])
def create_variable_for_generator(name, batch_size):
    truncation_psi_encode = 0.7
    layer_idx = np.arange(16)[np.newaxis, :, np.newaxis]
    ones = np.ones(layer_idx.shape, dtype=np.float32)
    coefs = tf.where(layer_idx < 8, truncation_psi_encode * ones, ones)
    dlatent_variable = tf.get_variable(
        'learnable_dlatents', 
        shape=(1, 16, 512), 
        dtype='float32', 
        initializer=tf.initializers.zeros()
    )
    dlatent_variable_trunc = tflib.lerp(dlatent_avg, dlatent_variable, coefs)
    return dlatent_variable_trunc

# Generation-from-disentangled-latents part
initial_dlatents = np.zeros((1, 16, 512))
Gs.components.synthesis.run(
    initial_dlatents,
    randomize_noise = True, # Turns out this should not be off ever for trying to lean dlatents, who knew
    minibatch_size = 1,
    custom_inputs = [
        partial(create_variable_for_generator, batch_size=1),
        partial(create_stub, batch_size = 1)],
    structure = 'fixed'
)

dlatent_variable = next(v for v in tf.global_variables() if 'learnable_dlatents' in v.name)
generator_output = tf.get_default_graph().get_tensor_by_name('G_synthesis_1/_Run/G_synthesis/images_out:0')
generated_image = tflib.convert_images_to_uint8(generator_output, nchw_to_nhwc=True, uint8_cast=False)
generated_image_uint8 = tf.saturate_cast(generated_image, tf.uint8)

# Loss part
vgg16 = VGG16(include_top=False, input_shape=(512, 512, 3))
perceptual_model = keras.Model(vgg16.input, vgg16.layers[9].output)
generated_img_features = perceptual_model(preprocess_input(generated_image, mode="tf"))
ref_img = tf.get_variable(
    'ref_img', 
    shape = generated_image.shape,
    dtype = 'float32', 
    initializer = tf.zeros_initializer()
)
ref_img_features = tf.get_variable(
    'ref_img_features', 
    shape = generated_img_features.shape,
    dtype = 'float32', 
    initializer = tf.zeros_initializer()
)
tf.get_default_session().run([ref_img.initializer, ref_img_features.initializer])
basic_loss = tf.losses.mean_squared_error(ref_img, generated_image)
perceptual_loss = tf.losses.mean_squared_error(ref_img_features, generated_img_features)

_D.run(np.zeros((1, 3, 512, 512)), None, custom_inputs = [
    lambda x: generator_output,
    partial(create_stub, batch_size = 1),
])
discriminator_output = tf.get_default_graph().get_tensor_by_name('D/_Run/D/scores_out:0')

# Attempt at making encoding better: Bias towards mean ("truncation loss", essentially)
dlatent_avg_full = dlatent_avg.reshape(-1, 512).repeat(16, axis = 0).reshape(-1, 16, 512)
input_loss = tf.losses.mean_squared_error(dlatent_variable, dlatent_avg_full)
combined_loss = input_loss + perceptual_loss

# We literally have a discriminator network, why not use it?
discriminator_loss = tf.nn.softplus(-discriminator_output)

# Gradient descend in latent space to something that is similar to the input image
def encode_image(image, iterations = 1024, learning_rate = 0.1, reset_dlatents = True, custom_initial_dlatents = None):
    # Get session
    sess = tf.get_default_session()
    
    # Gradient descent initial state
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    optimizer = tf.train.AdadeltaOptimizer(learning_rate = learning_rate)
    min_op = optimizer.minimize(perceptual_loss, var_list=[[dlatent_variable]])
    if reset_dlatents == True:
        if not custom_initial_dlatents is None:
            sess.run(tf.assign(dlatent_variable, custom_initial_dlatents.reshape(-1, 16, 512)))
        else:
            sess.run(tf.assign(dlatent_variable, initial_dlatents))
    
    # Generate and set reference image features
    ref_image_data = np.array(list(map(lambda x: (x.astype("float32")), [image])))
    image_features = perceptual_model.predict_on_batch(preprocess_input(ref_image_data, mode="tf"))  
    sess.run(tf.assign(ref_img_features, image_features))
    
    # Run
    for i in range(iterations):
        _, loss = sess.run([min_op, perceptual_loss])
        if i % 100 == 0:
            print("i: {}, l: {}".format(i, loss))
    
    # Generate image that actually goes with these dlatents for quick testing
    dlatents = sess.run(dlatent_variable)[0]
    generated_image = generate_images_from_dlatents(dlatents)
    
    return dlatents, generated_image

# Same as above but start with given dlatents and use plain MSE loss instead of vgg16
def finetune_image(dlatents, image, iterations = 32, learning_rate = 0.0001):
    # Get session and assign initial dlatents
    sess = tf.get_default_session()
    sess.run(tf.assign(dlatent_variable, np.array([dlatents])))
    
    # Set reference image
    ref_image_data = np.array(list(map(lambda x: (x.astype("float64")), [image])))
    sess.run(tf.assign(ref_img, ref_image_data))    
    
    # Gradient descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    min_op = optimizer.minimize(basic_loss, var_list=[[dlatent_variable]])
    
    for i in range(iterations):
        _, loss = sess.run([min_op, basic_loss])
        if i % 100 == 0:
            print("i: {}, l: {}".format(i, loss))

    # Generate image that actually goes with these latents for quick testing
    dlatents = sess.run(dlatent_variable)[0]
    generated_image = generate_images_from_dlatents(dlatents)
    
    return dlatents, generated_image

# Tune image in the direction of being considered more likely by the discriminator
def tune_with_discriminator(dlatents, iterations = 32, learning_rate = 1.0):
    # Get session and assign initial dlatents
    sess = tf.get_default_session()
    sess.run(tf.assign(dlatent_variable, np.array([dlatents])))
    
    # Gradient descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    min_op = optimizer.minimize(discriminator_loss, var_list=[[dlatent_variable]])
    
    for i in range(iterations):
        _, loss = sess.run([min_op, basic_loss])
        if i % 100 == 0:
            print("i: {}, l: {}".format(i, loss))
    
    return sess.run(dlatent_variable)[0]

# We have to do truncation ourselves, since we're not using the combined network
def truncate(dlatents, truncation_psi, maxlayer = 8):
    dlatent_avg = tf.get_default_session().run(Gs.own_vars["dlatent_avg"])
    layer_idx = np.arange(16)[np.newaxis, :, np.newaxis]
    ones = np.ones(layer_idx.shape, dtype=np.float32)
    coefs = tf.where(layer_idx < maxlayer, truncation_psi * ones, ones)
    return tf.get_default_session().run(tflib.lerp(dlatent_avg, dlatents, coefs))

# Generate image with disentangled latents as input
def generate_images_from_dlatents(dlatents, truncation_psi = 1.0, randomize_noise = True):
    if not truncation_psi is None:
        dlatents_trunc = truncate(dlatents, truncation_psi)
    else:
        dlatents_trunc = dlatents
        
    # Run the network
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    result_image = Gs.components.synthesis.run(
        dlatents_trunc.reshape((-1, 16, 512)),
        randomize_noise = randomize_noise,
        minibatch_size = 1,
        output_transform=fmt
    )[0]
    return result_image

# Sequence of learning steps while reducing lr followed by finetune
def encode_and_tune(image, iters_per_step = 1024):
    initial_latents = np.random.randn(1, Gs.input_shape[1])
    initial_dlatents = Gs.components.mapping.run(initial_latents, None)[0]
    dlatents_gen, image_gen = encode_image(image, iterations = iters_per_step, learning_rate = 100.0, custom_initial_dlatents = initial_dlatents)
    dlatents_gen2, image_gen2 = encode_image(image, iterations = iters_per_step, learning_rate = 10.0, reset_dlatents = False)
    dlatents_gen3, image_gen3 = encode_image(image, iterations = iters_per_step, learning_rate = 1.0, reset_dlatents = False)
    dlatents_gen4, image_gen4 = encode_image(image, iterations = iters_per_step, learning_rate = 0.1, reset_dlatents = False)
    dlatents_gen5, image_gen5 = encode_image(image, iterations = iters_per_step, learning_rate = 0.01, reset_dlatents = False)
    dlatents_gen6, image_gen6 = encode_image(image, iterations = iters_per_step, learning_rate = 0.001, reset_dlatents = False)
    dlatents_gen7, image_gen7 = finetune_image(dlatents_gen5, image, iterations = 128)
    return dlatents_gen7, image_gen7, dlatents_gen6


##
# 1. Just generate a neat interpolation video
##
# Pick latent vectors
#rnd = np.random.RandomState(5)
rnd = np.random
latents_a = rnd.randn(1, Gs.input_shape[1])
latents_b = rnd.randn(1, Gs.input_shape[1])
latents_c = rnd.randn(1, Gs.input_shape[1])

if os.path.exists("latents.npy"):
    latents_a, latents_b, latents_c = np.load("latents.npy")
np.save("latents.npy", np.array([latents_a, latents_b, latents_c]))


# "Ellipse around a point but probably a circle since it's 512 dimensions"
def circ_generator(latents_interpolate):
    radius = 40.0

    latents_axis_x = (latents_a - latents_b).flatten() / la.norm(latents_a - latents_b)
    latents_axis_y = (latents_a - latents_c).flatten() / la.norm(latents_a - latents_c)

    latents_x = math.sin(math.pi * 2.0 * latents_interpolate) * radius
    latents_y = math.cos(math.pi * 2.0 * latents_interpolate) * radius

    latents = latents_a + latents_x * latents_axis_x + latents_y * latents_axis_y
    return latents

# Generate images from a list of latents
def generate_from_latents(latent_list, truncation_psi):
    array_list = []
    image_list = []
    for latents in latent_list:
        # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=truncation_psi, randomize_noise=False, output_transform=fmt)
        array_list.append(images[0])
        image_list.append(PIL.Image.fromarray(images[0], 'RGB'))
        
    return array_list, image_list

def mse(x, y):
    return (np.square(x - y)).mean()

# Generate from a latent generator, keeping MSE between frames constant
def generate_from_generator_adaptive(gen_func):
    max_step = 1.0
    current_pos = 0.0
    
    change_min = 10.0
    change_max = 11.0
    
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    
    current_latent = gen_func(current_pos)
    current_image = Gs.run(current_latent, None, truncation_psi=0.5, randomize_noise=False, output_transform=fmt)[0]
    array_list = []
    
    while(current_pos < 1.0):
        array_list.append(current_image)
        
        lower = current_pos
        upper = current_pos + max_step
        current_pos = (upper + lower) / 2.0
        
        current_latent = gen_func(current_pos)
        current_image = images = Gs.run(current_latent, None, truncation_psi=0.5, randomize_noise=False, output_transform=fmt)[0]
        current_mse = mse(array_list[-1], current_image)
        
        while current_mse < change_min or current_mse > change_max:
            if current_mse < change_min:
                lower = current_pos
                current_pos = (upper + lower) / 2.0
            
            if current_mse > change_max:
                upper = current_pos
                current_pos = (upper + lower) / 2.0
                
            
            current_latent = gen_func(current_pos)
            current_image = images = Gs.run(current_latent, None, truncation_psi=0.5, randomize_noise=False, output_transform=fmt)[0]
            current_mse = mse(array_list[-1], current_image)
        print(current_pos, current_mse)
        
    return array_list


#array_list, _ = generate_from_latents(latent_list)
array_list = generate_from_generator_adaptive(circ_generator)
clip = moviepy.editor.ImageSequenceClip(array_list, fps=60)
clip.ipython_display()
#clip.write_videofile("out.mp4")


arrays, images = generate_from_latents([np.random.randn(1, Gs.input_shape[1])], 0.7)
images[0]

##
# 2. Encoding
##

# Load and cut and scale a bunch of data from the animefaces dataset
img_files = []
hair_cols = []
eye_cols = []
for in_dir in glob.glob("../../stylegan/animeface-character-dataset/thumb/*"):
    if not os.path.exists(in_dir + "/color.csv"):
        continue
    with open(in_dir + "/color.csv", 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            img_files.append(in_dir + "/" + row[0])
            hair_cols.append([row[1], row[2], row[3]])
            eye_cols.append([row[4], row[5], row[6]])
img_files = img_files[1::300]
hair_cols = hair_cols[1::300]
eye_cols = eye_cols[1::300]
#print(len(img_files))

image_arrays = []
for img_file in img_files:
    image_data = PIL.Image.open(img_file)
    image_size = min(image_data.width, image_data.height)
    image_data = image_data.crop((0, 0, image_size, image_size))
    image_data = image_data.resize((512, 512), PIL.Image.BILINEAR)
    image_array = np.array(image_data)
    image_arrays.append(image_array)

# Encode an image from there
dlatents_gen, image_gen = encode_and_tune(image_arrays[0])
im = PIL.Image.new('RGB', (1024, 512))
im.paste(PIL.Image.fromarray(image_arrays[0], 'RGB'), (0, 0))
im.paste(PIL.Image.fromarray(image_gen, 'RGB'), (512, 0))
im


# Re-encode a generated image 
# (mind: this is pointless for actual usage, if you're generating you can just 
# take the latents from the generation step). it's a nice check for encoding, though.
generated_ref = generate_from_latents([latents_a])[0][0]
dlatents_gen, image_gen = encode_and_tune(generated_ref, iters_per_step = 1024)
im = PIL.Image.new('RGB', (1024, 512))
im.paste(PIL.Image.fromarray(generated_ref, 'RGB'), (0, 0))
im.paste(PIL.Image.fromarray(image_gen, 'RGB'), (512, 0))
im

##
# 3. Modification
##
def generate_one():
    latents = rnd.randn(1, Gs.input_shape[1])
    dlatents = Gs.components.mapping.run(latents, None)[0]
    image = generate_images_from_dlatents(dlatents)
    return latents, dlatents, PIL.Image.fromarray(image, 'RGB')

def classify_image(generated_im):
    """
    there was a function here that used somebodies website to classify images for danbooru tags.
    since it's probably better to not have 10 people hit it, I removed it.
    the output .pkl for ~6k images is already pretty good to learn directions from and can be found here:
    
    https://drive.google.com/open?id=1_3Qvhj15bX_pETTENE7THQlfTjx1ghx3
    
    feel free to put any classifier here.
    """
    return([{"tag": likelihood_between_0_and_1}])

latent_list = []
dlatent_list = []
tag_list = []


# Generated and classify a bunch of images
while True:
    try:
        while True:
            temp_latents, temp_dlatents, temp_image = generate_one()
            temp_tags = classify_image(temp_image)
            latent_list.append(temp_latents)
            dlatent_list.append(temp_dlatents)
            tag_list.append(temp_tags)
            print("beep")
            
            if len(tag_list) % 500 == 0:
                with open("out_{}.pkl".format(len(tag_list)), 'wb') as f:
                    pickle.dump((latent_list, dlatent_list, tag_list), f)
                print("Wrote", "out_{}.pkl".format(len(tag_list)))
    except:
        print("nope")
        time.sleep(60)

with open("out_{}.pkl".format(len(tag_list)), 'wb') as f:
    pickle.dump((latent_list, dlatent_list, tag_list), f)
print("Wrote", "out_{}.pkl".format(len(tag_list)))


# Turn into features for learning directions
all_tags = collections.defaultdict(int)
for tags in tag_list:
    for tag in tags:
        all_tags[tag[0]] += 1
tags_by_popularity = sorted(all_tags.items(), key = lambda x: x[1], reverse = True)
eye_tags = list(filter(lambda x: x[0].endswith("_eyes"), tags_by_popularity))
hair_tags = list(filter(lambda x: x[0].endswith("_hair"), tags_by_popularity))

tag_binary_feats = {}
for tag, _ in tags_by_popularity:
    this_tag_feats = []
    for tag_list_for_dl in tag_list:
        this_dl_tag_value = 0.0
        for tag_for_dl, _ in tag_list_for_dl:
            if tag == tag_for_dl:
                this_dl_tag_value = 1.0
        this_tag_feats.append(this_dl_tag_value)
    tag_binary_feats[tag] = np.array(this_tag_feats)


# Learn directions for tags (some probably not very good)
def find_direction_binary(dlatents, targets):
    clf = LogisticRegression().fit(dlatents, targets)
    return clf.coef_.reshape((16, 512))

popular_tags = list(filter(lambda x: x[1] > 100, tags_by_popularity))
good_tags = list(filter(lambda x: (len(tag_list) - x[1]) > 1000, popular_tags))

dlatents_for_regression = np.array(dlatent_list).reshape(len(dlatent_list), 16*512)
tag_directions = {}
for i, (tag, _) in enumerate(good_tags):
    print("Estimating direction for", tag, "(", i, ")")
    tag_directions[tag] = find_direction_binary(dlatents_for_regression, tag_binary_feats[tag])

#with open("tag_dirs.pkl", 'wb') as f:
#    pickle.dump(tag_directions, f)
with open("tag_dirs.pkl", 'rb') as f:
    tag_directions = pickle.load(f)

# Do some modification
dlatents_gen = Gs.components.mapping.run(latents_a, None)[0]

im = PIL.Image.new('RGB', (512 * 5, 512 * 5))
for i in range(0, 5):
    for j in range(0, 5):
        factor_hair = (i / 4.0) * 2.0
        factor_eyes = (j / 4.0)
    
        dlatents_mod = copy.deepcopy(dlatents_gen)
        dlatents_mod += -tag_directions["blonde_hair"] * factor_hair + tag_directions["black_hair"] * factor_hair
        dlatents_mod += -tag_directions["green_eyes"] * factor_eyes + tag_directions["red_eyes"] * factor_eyes

        dlatents_mod_image = generate_images_from_dlatents(dlatents_mod, 0.7)
        im.paste(PIL.Image.fromarray(dlatents_mod_image, 'RGB'), (512 * i, 512 * j))

dlatents_gen = Gs.components.mapping.run(latents_c, None)[0]
dlatents_mod = copy.deepcopy(dlatents_gen)
dlatents_mod += -tag_directions["purple_hair"] * 1.0 + tag_directions["black_hair"] * 2.0 - tag_directions["green_hair"]
dlatents_mod += -tag_directions["blue_eyes"] * 1.0 + tag_directions["green_eyes"] * 1.0 - tag_directions["red_eyes"]
im = PIL.Image.new('RGB', (512 * 2, 512))
im.paste(PIL.Image.fromarray(generate_images_from_dlatents(dlatents_gen, 0.7), 'RGB'), (0, 0))
im.paste(PIL.Image.fromarray(generate_images_from_dlatents(dlatents_mod, 0.7), 'RGB'), (512, 0))
im

lock_updates

# Interactive modification!
hair_eyes_only = False
with open("tag_dirs.pkl", 'rb') as f:
    tag_directions = pickle.load(f)
    
tag_len = {}
for tag in tag_directions:
    tag_len[tag] = np.linalg.norm(tag_directions[tag].flatten())
    
mod_latents = np.load("mod_latents.npy")
dlatents_gen = Gs.components.mapping.run(mod_latents, None)[0]  
def modify_and_sample(psi, truncate_pre, truncate_post, **kwargs):
    if truncate_pre == True:
        dlatents_mod = truncate(copy.deepcopy(dlatents_gen), psi)
    else:
        dlatents_mod = copy.deepcopy(dlatents_gen)
        
    for tag in kwargs:
        dlatents_mod += tag_directions[tag] * kwargs[tag]
    value_widgets["psi"].value = str(round(psi, 2))
    
    for tag in kwargs:
        tag_value = round((np.dot(dlatents_mod.flatten(), tag_directions[tag].flatten()) / tag_len[tag]) - kwargs[tag], 2)
        value_widgets[tag].value = str(kwargs[tag]) + " | " + str(tag_value)
    
    display_psi = None
    if truncate_post == True:
        display_psi = psi
    display(PIL.Image.fromarray(generate_images_from_dlatents(dlatents_mod, truncation_psi = display_psi), 'RGB'))

psi_slider = widgets.FloatSlider(min = 0.0, max = 1.0, step = 0.01, value = 0.7, continuous_update = False, readout = False)
if hair_eyes_only:
    modify_tags = [tag for tag in tag_directions if "_hair" in tag or "_eyes" in tag or "_mouth" in tag]
else:
    with open("tags_use.pkl", "rb") as f:
        modify_tags = pickle.load(f)
    
modify_tags.append("realistic")
tag_widgets = {}
for tag in modify_tags:
    tag_widgets[tag] = widgets.FloatSlider(min = -3.0, max = 3.0, step = 0.01, continuous_update = False, readout = False)
all_widgets = []

sorted_widgets = sorted(tag_widgets.items(), key = lambda x: x[0])
sorted_widgets = [("psi", psi_slider)] + sorted_widgets
value_widgets = {}
for widget in sorted_widgets:
    label_widget = widgets.Label(widget[0])
    label_widget.layout.width = "140px"
    
    value_widget = widgets.Label("0.0+100.0")
    value_widget.layout.width = "150px"
    value_widgets[widget[0]] = value_widget
    
    tag_hbox = widgets.HBox([label_widget, widget[1], value_widget])
    tag_hbox.layout.width = "320px"
    
    all_widgets.append(tag_hbox)

refresh = widgets.Button(description="New Sample")
modify = widgets.Button(description="Mutate")

def new_sample(b):
    global mod_latents
    global dlatents_gen
    mod_latents = np.random.randn(1, Gs.input_shape[1])
    dlatents_gen = Gs.components.mapping.run(mod_latents, None)[0]  
    psi_slider.value += 0.00000000001
    #psi_slider.value -= 0.0000001
    
def mutate(b):
    global mod_latents
    global dlatents_gen
    mod_latents_add = np.random.randn(1, Gs.input_shape[1]) * 0.1
    mod_latents += mod_latents_add
    dlatents_gen = Gs.components.mapping.run(mod_latents, None)[0]  
    psi_slider.value += 0.00000000001
    #psi_slider.value -= 0.0000001

truncate_pre = widgets.ToggleButton(value=True, description='Truncate Pre')
truncate_post = widgets.ToggleButton(value=False, description='Truncate Post')
refresh.on_click(new_sample)
modify.on_click(mutate)

ui = widgets.Box(all_widgets + [refresh, modify, truncate_pre, truncate_post])
tag_widgets["psi"] = psi_slider

ui.layout.flex_flow = 'row wrap'
ui.layout.display = 'inline-flex'
tag_widgets["truncate_pre"] = truncate_pre
tag_widgets["truncate_post"] = truncate_post
out = widgets.interactive_output(modify_and_sample, tag_widgets)
display(ui, out)


with open("tags_use.pkl", "wb") as f:
    pickle.dump(tags, f)

tags.remove("pokemon_(creature)")