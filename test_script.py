"""Test script that creates database using a pickle file,
  loads the existing database, and has helper methods for 
  verifying and recognising face
"""

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
    enc1 = face_util.convert_to_embedding(single=True, imgpath1)
    enc2 = face_util.convert_to_embedding(single=True, imgpath2)
    
def who_it_is(img_path):
    # create database pickle file, if not exists
    if not os.path.exists('dbfile'):
        encodings = face_util.convert_to_embedding()
        store_data(encodings)

    # read the pickle file
    enc_list = load_data()

    test_img_enc = face_util.convert_to_embedding(single=True, img_path)
    dist_list = face_util.get_euclidean_dist_list(enc_list, test_img_enc)

    print (dist_list)
    flag = 0
    for item in dist_list:
        if item[1] <= 0.7:
            name = item[0].split('.')[0]
            print ("Welcome {}!!".format(name))
            flag = 1
            break
    if flag == 0:
        print ('No Record found in Database!!')
        print ('Intruder')


if __name__ == "__main__":

    #verify('roark.JPG', 'test.JPG')
    who_it_is('test.JPG')


