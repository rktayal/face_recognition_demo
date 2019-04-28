"""Test script that creates database using a pickle file,
  loads the existing database, and has helper methods for 
  verifying and recognising face
"""

import os
import sys
import pickle

from facenet_utils import FaceUtil

database_file_name = 'dbfile'
face_util = FaceUtil()

def store_data(encodings):
    with open('dbfile', 'wb') as f:
        pickle.dump(encodings, f)

def load_data():
    if os.path.exists('dbfile'):
        dbfile = open('dbfile', 'rb')
        db = pickle.load(dbfile)
        return db
    return None

def verify(imgpath1, imgpath2):
    enc1 = face_util.convert_to_embedding(single=True, img_path=imgpath1)
    enc2 = face_util.convert_to_embedding(single=True, img_path=imgpath2)
    dist = face_util.get_eucledian_dist(enc1, enc2)
    print ('distance is %1.4f' % dist)
    
def who_it_is(img_path):
    # create database pickle file, if not exists
    if not os.path.exists('dbfile'):
        encodings = face_util.convert_to_embedding()
        store_data(encodings)

    # read the pickle file
    enc_list = load_data()

    test_img_enc = face_util.convert_to_embedding(single=True, img_path=img_path)
    dist_list = face_util.get_eucledian_dist_list(enc_list, test_img_enc)

    #print (dist_list)
    flag = 0
    for item in dist_list:
        if item[1] <= 0.8:
            name = item[0].split('.')[0]
            print ("Welcome {}!!".format(name))
            flag = 1
            break
    if flag == 0:
        print ('No Record found in Database!!')
        print ('Intruder')


if __name__ == "__main__":

    #verify('./images/Roark.jpg', 'test.jpg')
    who_it_is('test.jpg')


