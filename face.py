#!/usr/bin/env python

import dlib
import numpy as np
import cv2
import imutils
import pickle
import os

class FaceRecognition:
    faces = {}
    ffaces = []

    def __init__(self):
       if not os.path.exists('faces.dat'):
           with open('faces.dat', 'w+') as f:
               pass

    def ConvertToBoxed(self, rect, shape):
        x = rect.left()
        y = rect.top()
        for i in shape:
            i[0] = i[0] - x
            i[1] = i[1] - y
        return shape

    def ConvertToFace(self, shape):
        coords = np.zeros((68, 2), dtype=int)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords
    def FacialSimilarity(self, a, b):
        c = []
        if len(a) == len(b):
            for i in range(len(a)):
                c.append(abs(a[i][0]-b[i][0])+abs(a[i][1]-b[i][1]))
            return sum(c)
        else:
            logging.error('Either of two faces were not calibrated to the 68 point model')
            return 100
    def NewFace(self, name, face):
        with open("faces.dat", "wb+") as f:
            self.faces[name] = face
            print("Learnt face", name)
            pickle.dump(self.faces,f)

    def Recognize(self, image):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("model.dat")
        image = imutils.resize(image, width=500)
        rects = detector(image, 1)
        if(len(rects)>0):
            for rect in rects:
                notSimilar=True
                shape = predictor(image, rect)
                shape = self.ConvertToFace(shape)
                shape = self.ConvertToBoxed(rect, shape)
                print('Current face', shape)
                with open("faces.dat", 'rb') as f:
                    try:
                        self.faces = pickle.load(f)
                        for x in self.faces.keys():
                            similarity = self.FacialSimilarity(self.faces[x], shape)
                            print('Similarity with', self.faces[x], 'is ', similarity)
                            if similarity < 200:
                                self.ffaces.append(x)
                                notSimilar = False
                        if notSimilar:
                            self.NewFace(input('Name: '), shape)
                    except EOFError:
                        print("Recreating faces.dat")
                        self.NewFace(input('Name: '), shape)
            return self.ffaces
        else:
            return []

print('faces in frame: ', FaceRecognition().Recognize(cv2.VideoCapture(0).read()[1]))
