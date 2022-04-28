# -*- coding: utf-8 -*-


#%%
import tensorflow
import streamlit as st
import pandas as pd
from PIL import Image, ImageOps
from keras.preprocessing import image
from scipy.optimize import minimize
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle


#%%
#img = Image.open('Brand.jpeg')
#st.image(img)


st.title('Simple App for Oil Spill Detection')
st.subheader('**You will interact with us!**')

st.write("""

         
First, go to the menu (left side) and select your chosen options.

Then, upload an image and choose the machine learning method.

Lastly, click in button Run to see the results.

This app detects oil spills in images. The process to classify the images is performed 
by feature extraction and machine learning models. Besides, the user can choose the 
image size and if he wants to see or note the corresponding feature map (output of 
the feature extraction process) on the screen.

The classification models that you can choose are: Support Vector Machine (SVM),
Extreme Gradient Boosting (XGB), and Logistic Regression (LR).

""")

#%%
#st.title('Oil Spill Detection')
#st.selectbox('Select the image size:', ['64x64', '256x256'])
#st.selectbox('Do you want to see the featuremap?', ['Yes, i do!', 'No, i do not!'])


st.sidebar.title('Menu')
inp_selec = st.sidebar.selectbox('Select image size:', 
                                 ['None', '64x64', '256x256'])

if inp_selec == 'None':
    featorno = st.sidebar.selectbox('Do you want to see the corresponding feature map?', 
                                    ['None','Yes, i do.', 'No, i don\'t.'])
    #st.title('You should input your image!')
    
elif inp_selec == '64x64':
    featorno = st.sidebar.selectbox('Do you want to see the corresponding feature map?', 
                                    ['None','Yes, i do.', 'No, i don\'t.'])
    #st.title('You should input your image!')

elif inp_selec == '256x256':
    featorno = st.sidebar.selectbox('Do you want to see the corresponding feature map?', 
                                    ['None', 'Yes, i do.', 'No, i don\'t.'])
    #st.title('You should input your image!')
    
file = st.file_uploader('', 
                        type=['jpeg', 'jpg', 'png'])

#%%

methods_names = ['SVM', 'XGB', 'LR']
methods = st.radio('Method', methods_names)
st.write("**The selected method is:**", methods)




#%%
t = [20.0, 50.0, 120.0, 45.0, 67.0, 24.0, 34.0, 23.0, 34.0, 23.0, 67.0, 78.0,
       34.0, 56.0, 56.0, 78.0]
def vero_qExp(x):
  q = x[0]
  eta = x[1]
  parte1 = 0
  n = len(t)
  for indice in t:
    parte1 = parte1 + (np.log(1 - ((1-q)*indice*(1/eta)))/(1-q))
  parte2 = (n*(np.log((2-q)/eta))) #; ipdb.set_trace() # Para debugar
  total = parte1 + parte2
  return -total

# Defining the image size
imgsize = 0
if inp_selec == 'None':
    imgsize = 64
elif inp_selec == '64x64':
    imgsize = 64
elif inp_selec == '256x256':
    imgsize = 256
    

def CDF_FE64(im1):
    count = 0
    for a in range(0, imgsize):
        for b in range(0, imgsize):
            if (im1[a][b]) == 0:
                count = count + 1
                im1[a][b] = 1
                
    histoCDF00 = []
    histoCDF01 = []
 
    for i in range(0, imgsize-3):
        for j in range(0, imgsize-3):
            x01 = [1.2, 300.0]
            t = [im1[i][j], im1[i][j+1], im1[i][j+2], im1[i][j+3], im1[i+1][j],
             im1[i+1][j+1], im1[i+1][j+2], im1[i+1][j+3], im1[i+2][j], im1[i+2][j+1],
             im1[i+2][j+2], im1[i+2][j+3], im1[i+3][j], im1[i+3][j+1], im1[i+3][j+2], im1[i+3][j+3]]
            sol0 = minimize(vero_qExp, x01, method='Nelder-Mead')
            #v_vero0 = sol0['fun']
            q = sol0['x'][0]
            eta = sol0['x'][1]
            med0 = np.mean(t)
            v_CDF0 = 1-((1-(1-q)*(med0/eta))**((2-q)/(1-q))) # Valor da CDF com a média de t
            histoCDF00.append(v_CDF0)
             
        histoCDF01.append(histoCDF00)
        histoCDF00 = []  
    histoCDF01 = np.array(histoCDF01)
    histoCDF01 = pd.DataFrame(histoCDF01)
    histoCDF01 = histoCDF01.fillna(0.01)
    histoCDF01 = np.array(histoCDF01)
    feature_vector = histoCDF01
    shap = feature_vector.shape
    print(feature_vector.shape)
        
    return shap, count, feature_vector

#%%

def CDF_FE256(im1):
    count = 0
    for a in range(0, imgsize):
        for b in range(0, imgsize):
            if (im1[a][b]) == 0:
                count = count + 1
                im1[a][b] = 1
                
    histoCDF00 = []
    histoCDF01 = []
    i = 0
    while i <= (imgsize-4):
        j = 0
        while j  <= (imgsize-4):
            x01 = [1.2, 300.0]
            t = [im1[i][j], im1[i][j+1], im1[i][j+2], im1[i][j+3], im1[i+1][j],
             im1[i+1][j+1], im1[i+1][j+2], im1[i+1][j+3], im1[i+2][j], im1[i+2][j+1],
             im1[i+2][j+2], im1[i+2][j+3], im1[i+3][j], im1[i+3][j+1], im1[i+3][j+2], im1[i+3][j+3]]
            sol0 = minimize(vero_qExp, x01, method='Nelder-Mead')
            #v_vero0 = sol0['fun']
            q = sol0['x'][0]
            eta = sol0['x'][1]
            med0 = np.mean(t)
            v_CDF0 = 1-((1-(1-q)*(med0/eta))**((2-q)/(1-q))) # Valor da CDF com a média de t
            histoCDF00.append(v_CDF0)
            j = j + 4
            
        i = i + 4   
        histoCDF01.append(histoCDF00)
        histoCDF00 = []  
        #i = i + 1
    histoCDF01 = np.array(histoCDF01)
    histoCDF01 = pd.DataFrame(histoCDF01)
    histoCDF01 = histoCDF01.fillna(0.01)
    histoCDF01 = np.array(histoCDF01)
    feature_vector = histoCDF01
    shap = feature_vector.shape
    print(feature_vector.shape)
        
    return shap, count, feature_vector

#%%

if st.button('Run'):
    if file is None:
        st.text('Please, upload an image file!')
    else:
        img = Image.open(file)
        st.image(img, width=600)
        img = img.resize((imgsize, imgsize))
        img_array = np.array(img)
        img_array = np.mean(img_array, -1)
        #st.text(img_array)
        shapee = img_array.shape
        st.text('The image size is:')
        st.text(shapee)
        st.text('Please, wait while the feature extraction process is completed!')
        if inp_selec == 'None':
            feat_vector = CDF_FE64(img_array)
        elif inp_selec == '64x64':
            feat_vector = CDF_FE64(img_array)
        elif inp_selec == '256x256':
            feat_vector = CDF_FE256(img_array)
        #feat_vector = CDF_FE64(img_array)
        arrayheat = feat_vector[2]
        sha = arrayheat.shape
        st.text('The heatmap size is:')
        st.text(sha)
        #st.text(feat_vector)
        if featorno == 'Yes, i do.':
            fig = sns.heatmap(data=arrayheat, yticklabels=False, xticklabels=False)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
        elif featorno == 'No, i don\'t.':
            st.text('Ok, we will not dysplay the correspondent heatmap!')
        elif featorno == 'None':
            st.text('Please, select Yes or Not!')
        arrayheat = arrayheat.reshape(len(arrayheat)*len(arrayheat))
        shapeimg = arrayheat.shape
        #st.text(shapeimg)
        arrayheat20 = []
        for i in range(0, 20):
            arrayheat20.append(arrayheat)
        arrayheat20 = np.array(arrayheat20)
        shape20 = arrayheat20.shape
        #st.text(shape20)
        pca = PCA(n_components=20)
        pca.fit(arrayheat20)
        arrayheat20 = pca.transform(arrayheat20)
        #shapeheat = arrayheat20.shape
        #st.text(shapeheat)
        arrayheat1 = arrayheat20[0]
        arrayheat1 = arrayheat1.reshape(1, 20)
        #shape1 = arrayheat1.shape
        #st.text(shape1)
    
    
    if imgsize == 64:
        if methods == 'SVM':
            load_SVM = pickle.load(open('predict_SVM.pkl', 'rb'))
            result_svm = load_SVM.predict(arrayheat1)
            prob = load_SVM.predict_proba(arrayheat1)
            #st.text(result_svm)
            #st.text(prob)
            if result_svm == 0.:
                #
                st.write("**Result using SVM**")
                st.text('The image has the following probability of do not present oil spills::')
                st.text(prob[0][0])
                st.text('Predicted class: \"0 - without oil spill\"')
                #st.text(result_svm)
                #img1 = Image.open('withoutoil.jpeg')
                #st.image(img1)
            else:
                #
                #st.text('The image has a high probability of presenting oil spills!')
                st.write("**Result using SVM**")
                st.text('The image has the following probability of present oil spills:')
                st.text(prob[0][1])
                st.text('Predicted class: \"1 - with oil spill\"')
                #st.text(result_svm)
                #img1 = Image.open('withoil.jpeg')
                #st.image(img1)
        elif methods == 'XGB':
            load_XGB = pickle.load(open('predict_XGB.pkl', 'rb'))
            result_xgb = load_XGB.predict(arrayheat1)
            prob = load_XGB.predict_proba(arrayheat1)
            if result_xgb == 0.:
                #
                st.write("**Result using XGB**")
                st.text('The image has the following probability of do not present oil spills::')
                st.text(prob[0][0])
                st.text('Predicted class: \"0 - without oil spill\"')
            else:
                #
                st.write("**Result using XGBoost**")
                st.text('The image has the following probability of present oil spills:')
                st.text(prob[0][1])
                st.text('Predicted class: \"1 - with oil spill\"')
                
        elif methods == 'LR':
            #
            load_LR = pickle.load(open('predict_LR.pkl', 'rb'))
            result_lr = load_LR.predict(arrayheat1)
            prob = load_LR.predict_proba(arrayheat1)
            if result_lr == 0.:
                #
                st.write("**Result using LR**")
                st.text('The image has the following probability of do not present oil spills::')
                st.text(prob[0][0])
                st.text('Predicted class: \"0 - without oil spill\"')
            else:
                #
                st.write("**Result using LR**")
                st.text('The image has the following probability of present oil spills:')
                st.text(prob[0][1])
                st.text('Predicted class: \"1 - with oil spill\"')
                
    if imgsize == 256:
        if methods == 'SVM':
            load_SVM = pickle.load(open('predict_SVM256.pkl', 'rb'))
            result_svm = load_SVM.predict(arrayheat1)
            prob = load_SVM.predict_proba(arrayheat1)
            #st.text(result_svm)
            #st.text(prob)
            if result_svm == 0.:
                st.write("**Result using SVM**")
                #st.text('The image has a probability of not presenting oil spills!')
                st.text('The image has the following probability of do not present oil spills:')
                st.text(prob[0][0])
                st.text('Predicted class: \"0 - without oil spill\"')
                #st.text(result_svm)
                #img1 = Image.open('withoutoil.jpeg')
                #st.image(img1)
            else:
                st.write("**Result using SVM**")
                #st.text('The image has a high probability of presenting oil spills!')
                st.text('The image has the following probability of present oil spills:')
                st.text(prob[0][1])
                st.text('Predicted class: \"1 - with oil spill\"')
                #st.text(result_svm)
                #img1 = Image.open('withoil.jpeg')
                #st.image(img1)
            
        elif methods == 'XGB':
            #
            load_XGB = pickle.load(open('predict_XGB256.pkl', 'rb'))
            result_xgb = load_XGB.predict(arrayheat1)
            prob = load_XGB.predict_proba(arrayheat1)
            if result_xgb == 0.:
                st.write("**Result using XGB**")
                st.text('The image has the following probability of do not present oil spills:')
                st.text(prob[0][0])
                st.text('Predicted class: \"0 - without oil spill\"')
            else:
                st.write("**Result using XGB**")
                st.text('The image has the following probability of present oil spills:')
                st.text(prob[0][1])
                st.text('Predicted class: \"1 - with oil spill\"')
                
        elif methods == 'LR':
            #
            load_LR = pickle.load(open('predict_LR256.pkl', 'rb'))
            result_lr = load_LR.predict(arrayheat1)
            prob = load_LR.predict_proba(arrayheat1)
            if result_lr == 0.:
                #
                st.write("**Result using LR**")
                st.text('The image has the following probability of do not present oil spills:')
                st.text(prob[0][0])
                st.text('Predicted class: \"0 - without oil spill\"')
            else:
                #
                st.write("**Result using LR**")
                st.text('The image has the following probability of present oil spills:')
                st.text(prob[0][1])
                st.text('Predicted class: \"1 - with oil spill\"')
        
else:
    st.write("**Click in button Run to see the results**")
        
    
#%%



















    