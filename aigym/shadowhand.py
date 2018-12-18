import gym
import random
import png
import numpy
import time
import cv2

WIDTH = 1024
HEIGHT = 1024

env = gym.make('HandManipulateBlock-v0')
env.reset()
env.env.render()
env.env.viewer._hide_overlay = True
pngWriter = png.Writer(width=WIDTH,height=HEIGHT)

# here is waiting loop 
for i in range(50):
    env.render()
    time.sleep(0.1)

for i in range (3000):
    env.env.viewer.cam.distance = 0.5
    env.env.viewer.cam.azimuth = 90.
    env.env.viewer.cam.elevation = -45.
    name = '1024/out' + str(i)
    with open(name+".png",'wb') as f:
        action = numpy.random.randn(20)
        obs, reward, done, info = env.env.step(action)
        joints = obs['observation'][0:24]
        image = env.render('rgb_array')
        image = image[ :, 415:1435, :]
        image = cv2.resize(image, (WIDTH,HEIGHT))              
        numpy.save(name,image)
        numpy.save(name+"-joints",joints)
        pngWriter.write(f, numpy.reshape(image, (-1, WIDTH*3)))


         
