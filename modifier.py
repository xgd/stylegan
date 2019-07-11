#@title Interactive Waifu Modification { run: "auto" }
if (not 'init_done' in globals()) or init_done == False:
    # Download data / libraries
    # Thanks, StackOverflow https://stackoverflow.com/questions/49685924/extract-google-drive-zip-from-google-colab-notebook
    # idk if too many downloads will get me dinged by g drive so, if this stops working, well.
    import requests
    import os

    def download_file_from_google_drive(id, destination):
        def get_confirm_token(response):
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    return value

            return None

        def save_response_content(response, destination):
            CHUNK_SIZE = 32768

            with open(destination, "wb") as f:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)

        URL = "https://docs.google.com/uc?export=download"

        session = requests.Session()

        response = session.get(URL, params = { 'id' : id }, stream = True)
        token = get_confirm_token(response)

        if token:
            params = { 'id' : id, 'confirm' : token }
            response = session.get(URL, params = params, stream = True)
        save_response_content(response, destination)  

    if not os.path.exists("download.zip"):
        download_file_from_google_drive("1Ir3bawyYhj9G5I0HBox9eA2sip_T7bj1", "download.zip")
        !unzip download.zip

    import os
    import pickle
    import numpy as np
    import numpy.linalg as la
    import PIL.Image
    import PIL.ImageSequence

    import dnnlib
    import dnnlib.tflib as tflib
    from IPython.display import display, clear_output
    import math
    import glob
    import csv
    from functools import partial
    import time
    import collections

    import tensorflow as tf

    import keras
    from keras.applications.vgg16 import VGG16, preprocess_input

    import re
    import copy

    from ipywidgets import interact, interactive, fixed, interact_manual
    import ipywidgets as widgets

    import io
    from matplotlib import pyplot as plt

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
    # Unfortunately this works only Kind Of Okay
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

    # Load up tags and tag directions
    with open("tag_dirs.pkl", 'rb') as f:
        tag_directions = pickle.load(f)

    tag_len = {}
    for tag in tag_directions:
        tag_len[tag] = np.linalg.norm(tag_directions[tag].flatten())

    mod_latents = np.load("mod_latents.npy")
    dlatents_gen = Gs.components.mapping.run(mod_latents, None)[0]

    with open("tags_use.pkl", "rb") as f:
        modify_tags = pickle.load(f)

    """
    Generates the UI code for google colab
    for tag in modify_tags:
        tag_var = tag
        if tag_var == "+_+":
           tag_var = "plus_eyes"
        if tag_var == ":d":
           tag_var = "big_smile"
        print(tag_var + " = 0.0 #" + "@param {type:\"slider\", min: -2.0, max: 2.0, step: 0.01}") 
        print("tag_factors['" + tag + "'] = " + tag_var)
    """
    clear_output()
    
init_done = True

tag_factors = {}

# Interactive modification (google colab version)
psi = 0.67 #@param {type:"slider", min: 0.0, max:1.0, step: 0.01}

plus_eyes = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['+_+'] = plus_eyes
big_smile = 0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors[':d'] = big_smile
bangs = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['bangs'] = bangs
black_hair = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['black_hair'] = black_hair
blonde_hair = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['blonde_hair'] = blonde_hair
blue_eyes = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['blue_eyes'] = blue_eyes
blue_hair = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['blue_hair'] = blue_hair
blunt_bangs = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['blunt_bangs'] = blunt_bangs
blurry = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['blurry'] = blurry
blush = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['blush'] = blush
bow = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['bow'] = bow
braid = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['braid'] = braid
brown_eyes = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['brown_eyes'] = brown_eyes
brown_hair = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['brown_hair'] = brown_hair
closed_eyes = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['closed_eyes'] = closed_eyes
closed_mouth = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['closed_mouth'] = closed_mouth
collared_shirt = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['collared_shirt'] = collared_shirt
comic = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['comic'] = comic
day = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['day'] = day
double_bun = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['double_bun'] = double_bun
earrings = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['earrings'] = earrings
emphasis_lines = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['emphasis_lines'] = emphasis_lines
eyebrows_visible_through_hair = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['eyebrows_visible_through_hair'] = eyebrows_visible_through_hair
eyelashes = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['eyelashes'] = eyelashes
eyes = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['eyes'] = eyes
green_eyes = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['green_eyes'] = green_eyes
green_hair = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['green_hair'] = green_hair
hair_between_eyes = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['hair_between_eyes'] = hair_between_eyes
hair_flaps = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['hair_flaps'] = hair_flaps
hair_ornament = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['hair_ornament'] = hair_ornament
hairclip = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['hairclip'] = hairclip
hat = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['hat'] = hat
heart = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['heart'] = heart
jewelry = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['jewelry'] = jewelry
lips = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['lips'] = lips
long_hair = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['long_hair'] = long_hair
monochrome = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['monochrome'] = monochrome
necktie = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['necktie'] = necktie
one_eye_closed = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['one_eye_closed'] = one_eye_closed
open_mouth = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['open_mouth'] = open_mouth
orange_hair = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['orange_hair'] = orange_hair
outdoors = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['outdoors'] = outdoors
parody = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['parody'] = parody
peeking_out = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['peeking_out'] = peeking_out
pink_hair = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['pink_hair'] = pink_hair
pointy_ears = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['pointy_ears'] = pointy_ears
portrait = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['portrait'] = portrait
profile = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['profile'] = profile
purple_eyes = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['purple_eyes'] = purple_eyes
purple_hair = 0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['purple_hair'] = purple_hair
realistic = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['realistic'] = realistic
red_eyes = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['red_eyes'] = red_eyes
ribbon = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['ribbon'] = ribbon
school_uniform = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['school_uniform'] = school_uniform
serafuku = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['serafuku'] = serafuku
shirt = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['shirt'] = shirt
short_hair = 0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['short_hair'] = short_hair
sidelocks = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['sidelocks'] = sidelocks
silver_hair = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['silver_hair'] = silver_hair
simple_background = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['simple_background'] = simple_background
sky = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['sky'] = sky
smile = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['smile'] = smile
sparkling_eyes = 0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['sparkling_eyes'] = sparkling_eyes
sweat = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['sweat'] = sweat
tail = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['tail'] = tail
tongue = -0.94 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['tongue'] = tongue
tongue_out = 0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['tongue_out'] = tongue_out
uvula = 0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['uvula'] = uvula
visor = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['visor'] = visor
white_background = 0.0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['white_background'] = white_background
white_hair = 0 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['white_hair'] = white_hair
yellow_eyes = -0.48 #@param {type:"slider", min: -2.0, max: 2.0, step: 0.01}
tag_factors['yellow_eyes'] = yellow_eyes

truncate_pre = True #@param {type:"boolean"}
truncate_post = True #@param {type:"boolean"}
new_sample = False #@param {type:"boolean"}
mutate = False #@param {type:"boolean"}

if new_sample:
    mod_latents = np.random.randn(1, Gs.input_shape[1])
    dlatents_gen = Gs.components.mapping.run(mod_latents, None)[0] 
    
if mutate:
    mod_latents_add = np.random.randn(1, Gs.input_shape[1]) * 0.2
    mod_latents += mod_latents_add
    dlatents_gen = Gs.components.mapping.run(mod_latents, None)[0] 
    
if truncate_pre == True:
    dlatents_mod = truncate(copy.deepcopy(dlatents_gen), psi)
else:
    dlatents_mod = copy.deepcopy(dlatents_gen)

for tag in tag_factors:
    dlatents_mod += tag_directions[tag] * tag_factors[tag]
    
display_psi = None
if truncate_post == True:
    display_psi = psi
clear_output()    
display(PIL.Image.fromarray(generate_images_from_dlatents(dlatents_mod, truncation_psi = display_psi), 'RGB'))
