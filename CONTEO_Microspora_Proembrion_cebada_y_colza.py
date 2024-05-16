#!/usr/bin/env python3
#
# Reconocimiento de proembriones y microsporas en imágenes de microscopía en cultivos de COLZA y CEBADA.
# IMPORTANTE: INPUT (carpeta con imágenes) con subdirectorios anidados (minimo 2 niveles) que indiquen experimento> concentración > réplica
# COLZA : Detección de círculos y recuento en grupos según radio predicho en clasificacion (KMEANS).
# CEBADA : Generación de máscaras con regiones de interés, análisis y clasificación en función del  área y excentricidad de las elipses incluidas en dichas regiones
# Recuento de clasificaciones, representación en diagrama de barras

# ANOVA repeated measures (one-way), TUKEY post-hoc

# Grupo IP Pilar Sánchez-Testillano
# Natalia García
# NGS/MAYO2023
# 
# Este codigo fue programado para:
# python 3.6.5
# pillow 5.4.1
# opencv 3.4.2.17
#
# Parámetros particulares
# COLZA HOUGH TRANSFORM : dp=1.3, param1=50, param2=15, minRadius=5, maxRadius=12
# CEBADA : número de dilataciones / erosiones, kernel 

from math import nextafter
import math
import cv2
import numpy as np
from PIL import Image, ImageTk
from skimage.measure import regionprops_table
from skimage.morphology import closing
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage import morphology
from sklearn.mixture import BayesianGaussianMixture
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# gui
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

# directory checks, access
import os
# data wrangling
import pandas as pd
# plot final detected and classified colors 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# procesado y clasificacion de colores medio
from sklearn import preprocessing
from sklearn.cluster import KMeans



# guardar y cargar modelos
import pickle

# remove size limitation
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))








def entrenar_kmeans():
    
    # Crea carpeta de resultados general
    # ruta_guardar = input("Escribe el nombre del directorio \\n-----------------------")
    #ruta_guardar = "C:\\\\Users\\\\Usuario\\\\Desktop\\\\Fotos para contar\\\\output_segundo_modelo"
    ruta_guardar = filedialog.askdirectory(title="Directorio para guardar el output del modelo")

    try:
        os.mkdir(ruta_guardar) #
    except:
        1
    # pregunta por directorio y saca imagenes
    dir_path = filedialog.askdirectory(title="Directorio con fotos para entrenar KMEANS")


    # coge solo jpg o tif
    onlyfiles = [f for f in os.listdir(dir_path) if os.path.isfile("\\\\".join([dir_path, f])) and ".jpg" in f or ".tif" in f and "Merged" not in f ]

    
    tag_circle = 0
    num_circles = 0

     # df_color_circles : guarda infromacion sobre los valores rgb de todos los pixeles separados por cada circulo, se preprocesa para hacer clustering no supervisado
    # df_info_circles : guarda informacion sobre las coordenadas de los circulos y en qué imagen aparecen, se usará para anotar las imágenes
       
    df_info_circles = pd.DataFrame(columns=["Ctag", "x", "y", "r", "filename"])

    #####################################################################################################################
    # PRIMERA PARTE : ANALISIS DE  IMAGEN
    # PRIMERA ITERACION POR FICHEROS - DETECTA CIRCULOS, GUARDA INFORMACIÓN ACERCA DE METRICAS
    # MÁS ADELANTE SE INCORPORARÁ INFORMACIÓN ACERCA DEL - COLOR DOMINANTE DE LOS 3 CLUSTERS, Nº DE CIRCULOS EN CADA CLUSTER

    # MEDIA RGB IMAGEN - VALOR APROXIMADO DEL BACKGROUND
    # RADIO DE CIRCULOS SIN THRESHOLD INFERIOR
    # RADIO DE CIRCULOS CRIBADOS
    # NUMERO DE CIRCULOS CRIBADO

    for filename in onlyfiles:

        radios_circulo_np = []
        radios_circulo_p = []
        

        if filename:
            

            # Leer la imagen
            image = cv2.imread(dir_path+"\\\\"+filename, 1) #
            # Cropping an image - quita banner de la escala de la foto, que entromete a la hora de entrenar el modelo
            image = image[:-85, :]

            print(f"Performing processing on file : {filename}\\n")

            try:
                # Convertir a escala de grises
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #
                
                # # Aplicar un suavizado para reducir el ruido
                # # #5, 5
                # blurred = cv2.GaussianBlur(gray, (9, 9), 0) #
                
                # Detección de bordes utilizando el detector de Canny
                #umin 50
                edges = cv2.Canny(gray, 60, 100)
                
                # Detección de círculos utilizando HoughCircles
                #dp=1.3, minDist=60, param1=55, param2=20, minRadius=30, maxRadius=60
                #dp=1.5, minDist=35, param1=55, param2=30, minRadius=20, maxRadius=60
                circles = cv2.HoughCircles(
                    edges, cv2.HOUGH_GRADIENT,  dp=1.5, minDist=35, param1=55, param2=30, minRadius=10, maxRadius=70
                )
                
                
                # iterando por cada circulo detectado
                if circles is not None:
                    circles = np.uint16(np.around(circles))

                    for circle in circles[0, :]:
                        x, y, r = circle
                        # guardar informacion
                        radios_circulo_np.append(r)

                        if r < 70:  # Solo destacar círculos menores de 40 píxeles
                            
                            tag_circle += 1
                            # guardar informacion
                            x, y, r = circle

                            radios_circulo_p.append(r)
                            # nº de circulos detectados
                            num_circles += 1

                            # anexar a df 
                            df_info_circles = pd.concat([df_info_circles, pd.DataFrame([[tag_circle, x, y, r, filename]], columns =["Ctag", "x", "y", "r", "filename"])], ignore_index = True, axis = 0 )


                
                df_info_foto = pd.concat([df_info_foto, pd.DataFrame([output_estadisticas_foto], columns=columnas_excel)])
                num_circles = 0
                
            except:
                pass
            
            del image
    ###################################################################################
    # SEGUNDA PARTE - ESTANDARIZACION DE VALORES RGB PARA LOS CIRCULOS Y CLUSTERIZACIÓN
    
    # 1) Estandarizacion con Standard Scaler
    # Asume misma desviacion por canal de colores en los circulos - decisión logica si proceden crops de misma foto

    print("Scaling color channels in cell rois...\\n")


    df_circr = df_info_circles[["r", "Ctag"]]
    
    # hacer histograma de valores de r entre todas las imagenes
    fig = plt.figure()
    sns.distplot(df_circr[["r"]])
    plt.show()
    plt.savefig("C:\\\\Users\\\\Usuario\\\\Downloads\\\\hist_r.png", bbox_inches = "tight")


    scaler = preprocessing.StandardScaler().fit(df_circr[["r"]])
    df_circr["r_scaled"] = scaler.transform(df_circr[["r"]])

    # sacar promedio rgb roi para que todas las observaciones tengan la misma dimension
    df_to_cluster = df_circr[["r_scaled", "Ctag"]]

    
    ##########################################################################
    # TERCERA PARTE : ENTRENAR y guardar modelo de  KMEANS
    # KMEANS

    print("Performing kmeans...\\n")

    km = KMeans( n_clusters = 2 ) # 2 clasificaciones
    km.fit(df_to_cluster[["r_scaled"]])
    df_to_cluster['label'] = km.predict(df_to_cluster[["r_scaled"]])
    
    
    # save model
    
    pickle.dump(km, open(f"{ruta_guardar}\\\\kmeans_bnap.pkl", 'wb')) #Saving the model
    
    # Muestra un mensaje de confirmación
    tk.messagebox.showinfo("Modelo de KMEANS guardado", f"La imagen y estadisticas de la imagen se ha guardado en el directorio {ruta_guardar}")
    
    del df_to_cluster
    return
    
    
    










    
    
def predecir_cl_colza():

    columnas_excel =["experiment", "conc", "replica", "filename"]
    df_info_foto = pd.DataFrame(columns = columnas_excel )         
    df_info_circles = pd.DataFrame(columns=["Ctag", "x", "y", "r", "filename"])

    # pregunta por directorio y saca imagenes
    tk.messagebox.showinfo("INSTRUCCIONES PROGRAMA - input", f"Elija la carpeta anidada con imagenes a contar\n\n- Las imagenes deben de estar incluidas en dos a tres niveles de carpetas, que describan la estructura del experimento\n\n- Deben describir : \n\n\t> Factor principal (pej. molecula usada) >\n\t\t> Factor secundario (concentracion)\n\t\t\t> réplicas (*con imágenes)\n\n\t\t\tó\n\n\t> Factor principal (molecula usada)\n\t\t> réplicas (*carpetas con imágenes) \n\n- La carpeta de control no hace falta que tenga una segunda carpeta anidada\n\nIMPORTANTE : Si no se presentan los datos de entrada así, no se procesarán bien las imagenes ni se realizarán bien los test de hipótesis")
    
    dir_path = filedialog.askdirectory(title="Directorio con fotos para conteo").replace("/", "\\\\")
    df_todas_fotos = pd.DataFrame(columns=["filename", "Proembryo", "Vacuolated Microspore"])
    

    # CARGAR MODELO kmeans Y ASIGNAR INFORMACIÓN DE A QUÉ LABEL CORRESPONDE CADA CLUSTER
    #modelfile = filedialog.askopenfilename(title="Archivo .pkl con modelo KMEANS", filetypes =[('Pickle Files', '*.pkl')])
    
    modelfile = "C:\\Users\\Usuario\\Desktop\\Cuantificacion proembriones\\kmeans_bnap.pkl"
    

    if modelfile is not None:
        # load the model from disk
        loaded_model = pickle.load(open(modelfile, 'rb'))

    # Si el cluster tiene una media - centroide de radio mayor, se le etiqueta como proembrión, si no, como microspora

    dict_l_map = dict(zip(loaded_model.labels_, [None, None]))
    for label_cluster in loaded_model.labels_:
        if label_cluster == np.argmax(np.array(loaded_model.cluster_centers_)):
            dict_l_map[label_cluster] = "Proembryo"
        else:
            dict_l_map[label_cluster] = "Vacuolated Microspore"

    print(dict_l_map)
    
    # Crea carpeta de resultados general
    tk.messagebox.showinfo("INSTRUCCIONES PROGRAMA - output", f"Elija la carpeta donde quiere que se volquen los resultados (imágenes, visualizaciones y test estadísticos)")
    ruta_guardar = filedialog.askdirectory(title="Directorio para guardar el output de fotos con la predicción").replace("/", "\\\\")
    print(ruta_guardar+"\\n--------------\\n--------------\\n------------")
    try:
        os.mkdir(ruta_guardar) #
    except:
            1
            
            
    
    from glob import glob
    
    experiments = glob(dir_path+"\\\\*", recursive=True)
    
    experiments = [os.path.basename(directory_parent) for directory_parent in experiments ]
    
    print(experiments)
    
    for experiment in experiments:
        
        for subdirpath, subdirs, files in os.walk(dir_path+"\\\\"+experiment):               
            
            
            if len(files) != 0:
                
                conc = os.path.basename(os.path.dirname(subdirpath))
                
                repname = os.path.basename(subdirpath)
                
                if "Metadata" not in os.path.basename(subdirpath):
                    
        
                    
                        #####################################################################################################################
                        # PRIMERA PARTE : ANALISIS DE  IMAGEN
                        # PRIMERA ITERACION POR FICHEROS - DETECTA CIRCULOS, GUARDA INFORMACIÓN ACERCA DE METRICAS
                        # MÁS ADELANTE SE INCORPORARÁ INFORMACIÓN ACERCA DEL - COLOR DOMINANTE DE LOS 3 CLUSTERS, Nº DE CIRCULOS EN CADA CLUSTER
        
                        # MEDIA RGB IMAGEN - VALOR APROXIMADO DEL BACKGROUND
                        # RADIO DE CIRCULOS SIN THRESHOLD INFERIOR
                        # RADIO DE CIRCULOS CRIBADOS
                        # NUMERO DE CIRCULOS CRIBADO
                        
                    
                    
                    for filename in files:
                        
                        print(f"{dir_path}\\\\{experiment}_{conc}_{repname}\\\\{filename}")
                        radios_circulo_np = []
                        radios_circulo_p = []
                        
                        tag_circle = 0
                        
        
                        if filename:
                            
                            try:
                                if conc == experiment:
                                    bool_noconc = True
                                    # Leer la imagen
                                    image = cv2.imdecode(np.fromfile(f"{dir_path}\\\\{experiment}\\\\{repname}\\\\{filename}", dtype=np.uint8), cv2.IMREAD_UNCHANGED )
                                    #image = cv2.imread(f"{dir_path}\\\\{experiment}\\\\{repname}\\\\{filename}", 1) #
                                else:
                                    bool_noconc = False
                                    # Leer la imagen
                                    image = cv2.imdecode(np.fromfile(f"{dir_path}\\\\{experiment}\\\\{conc}\\\\{repname}\\\\{filename}", dtype=np.uint8), cv2.IMREAD_UNCHANGED )
                                    #image = cv2.imread(f"{dir_path}\\\\{experiment}\\\\{conc}\\\\{repname}\\\\{filename}", 1) #
                                # Cropping an image - quita banner de la escala de la foto, que entromete a la hora de entrenar el modelo
                                image = image[:-85, :]
            
                                # Convertir a escala de grises
                                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #
                            
                                
                                # ECUALIZACION
                                #ecualizada = cv2.equalizeHist(gray) #
                                
                                # # # Aplicar un suavizado para reducir el ruido
                                blurred = cv2.GaussianBlur(gray, (5, 5), 0) #
                                
                                # Detección de bordes utilizando el detector de Canny
                                edges = cv2.Canny(blurred, 50, 100)
                                
                                # Detección de círculos utilizando HoughCircles
                                #dp=1.3, minDist=60, param1=55, param2=20, minRadius=30, maxRadius=60
                                #dp=1.5, minDist=35, param1=55, param2=30, minRadius=20, maxRadius=60
                                #dp=2, minDist=40, param1=55, param2=30, minRadius=10, maxRadius=70
                                circles = cv2.HoughCircles(
                                    edges, cv2.HOUGH_GRADIENT, dp=1.4, minDist=40, param1=55, param2=30, minRadius=10, maxRadius=60
                                )
                                
                                
                                print("Procesando imagen : "+filename+"\\n")
                            
                                # iterando por cada circulo detectado
                                if circles is not None:
                                    circles = np.uint16(np.around(circles))
            
                                    for circle in circles[0, :]:
                                        x, y, r = circle
                                        # guardar informacion
                                        radios_circulo_np.append(r)
            
                                        if r < 40:  # Solo destacar círculos menores de 40 píxeles
                                            
                                            tag_circle += 1
                                            ctag_i = ";".join([str(tag_circle), repname, experiment, conc, filename])
                                            
                                            filename_excel = ";".join([experiment, conc, repname, filename ])
                                            
                                            # guardar informacion
                                            x, y, r = circle
            
                                            radios_circulo_p.append(r)
                                            
                                            
                                            # anexar a df 
                                            df_info_circles = pd.concat([df_info_circles, pd.DataFrame([[ctag_i, x, y, r, filename_excel]], columns =["Ctag", "x", "y", "r", "filename"])], ignore_index = True, axis = 0 )
                                            output_estadisticas_foto  = [experiment, conc, repname, filename_excel ]
                                            df_info_foto = pd.concat([df_info_foto, pd.DataFrame([output_estadisticas_foto], columns=columnas_excel)])
                            except:
                                continue
            
                        
    ###################################################################################
    # SEGUNDA PARTE - ESTANDARIZACION de radios, prediccion de clase PE/VM y volcado de frecuencias en excel
    
    # 1) Estandarizacion con scikitlearn StandardScaler
    # Escalar todos los radios de todas las imagenes a una misma dstr de radio.
    
    df_circr = df_info_circles[["x", "y", "r", "Ctag", "filename"]]
    
    scaler = preprocessing.StandardScaler().fit(df_circr[["r"]])
    df_circr["r_scaled"] = scaler.transform(df_circr[["r"]])

    
    # # KMEANS PREDICT WITH LOADED MODEL
    print("Performing kmeans...\\n")
    df_circr['label'] = loaded_model.predict(df_circr[["r_scaled"]])
    
    print("Counting cell frequencies...\\n")
    # ## inner join coordenadas radio y clasificacion
    # # La variable df_annotated tendrá información acerca de en qué imagen se situan los circulos, sus coordenadas y su clasificacion
    df_annotated = df_circr[["Ctag", "label", "filename"]]
    
    # # Anotar frecuencia de cada label en cada foto
    df_freq = df_annotated.groupby(by=["filename", "label"], as_index=False).count()
    df_freq = df_freq.drop_duplicates()
    
    print("Assigning labels to human-readable label tags...\\n")
    df_freq["interpretable_label"] = [dict_l_map[label_cluster] for label_cluster in df_freq["label"].values.tolist()]
    df_freq.rename(columns={"Ctag":"count"}, inplace=True)
    df_freq = df_freq[["filename", "interpretable_label", "count"]]
    

    print("Reformatting cell frequency dataframe..\\n")
    df_freq_2 = df_freq.pivot(values="count", columns='interpretable_label', index="filename").reset_index(names="filename")



    df_freq_2 = df_freq_2.drop_duplicates()
    df_freq_2 = df_freq_2.fillna(0)
    viz_tinker(df_freq_2, ruta_guardar, bool_noconc)
    
    
    
    ###################################################################################
    # TERCERA PARTE - ANOTACION DE IMAGENES EN RUTA DE GUARDADO con el dataframe df_annotated
    
    for experiment in experiments:
        
        for subdirpath, subdirs, files in os.walk(dir_path+"\\\\"+experiment):               
            
            
            if len(files) != 0:
                
                conc = os.path.basename(os.path.dirname(subdirpath))
                
                repname = os.path.basename(subdirpath)
                
                if "Metadata" not in os.path.basename(subdirpath):
                    # try:
                    #os.mkdir(ruta_guardar+f"\\\\{experiment}")
                    os.mkdir(ruta_guardar+f"\\\\{experiment}_{conc}_{repname}") #
                    # except:
                    #     1
        
                    
                    
                    for filename in files:
                        
                        print("Annotating filename "+filename)
                        filename_try =  ";".join([experiment, conc, repname, filename ])
                        
                        try:
                            print("Reading images...")
                            if conc == experiment:
                                # Leer la imagen
                                image = cv2.imdecode(np.fromfile(f"{dir_path}\\\\{experiment}\\\\{repname}\\\\{filename}", dtype=np.uint8), cv2.IMREAD_UNCHANGED )
                                #image = cv2.imread(f"{dir_path}\\\\{experiment}\\\\{repname}\\\\{filename}", 1) #
                            else:
                                # Leer la imagen
                                image = cv2.imdecode(np.fromfile(f"{dir_path}\\\\{experiment}\\\\{conc}\\\\{repname}\\\\{filename}", dtype=np.uint8), cv2.IMREAD_UNCHANGED )
                                #image = cv2.imread(f"{dir_path}\\\\{experiment}\\\\{conc}\\\\{repname}\\\\{filename}", 1) #
                            image = image[:-85, :]
                            
                            print("Getting cell coordinates...")
                            df_tmp = df_circr[df_circr["filename"] == filename_try][["x", "y", "r", "label"]]
                            
                            print("Getting cell coordinates...")
                            # iterar por circulo anotado
                            for row_info_circle in df_tmp.values.tolist():
            
                                x, y, r, label_col  = row_info_circle 
            
                                # 1) ANOTAR CIRCULOS
                                # más adelante pongo el color del v de cada centroide
            
                                if label_col == 0:
                                    cv2.circle(image, (x, y), r+20, (0, 0, 255), 2)  # Dibujar círculos rojos en la imagen
                                elif label_col == 1:
                                    cv2.circle(image, (x, y), r+20, (0, 255, 0), 2)  # Dibujar círculos verdes en la imagen
                            
                            print("Getting cell class frequencies...")
                            contador_PE = df_freq_2[df_freq_2["filename"] == filename_try][["Proembryo"]].to_string()
                            contador_VM = df_freq_2[df_freq_2["filename"] == filename_try][["Vacuolated Microspore"]].to_string()
                            
                            
                            print("Printing cell class frequencies...")
                            # 2) ANOTAR FRECUENCIA DE CLUSTERS
                            cv2.putText(image, "PROEMBRIONES - ROJO : "+str(contador_PE), (20, 50) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                            cv2.putText(image, "MICROESPORAS - VERDE : "+str(contador_VM), (20, 90) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                            
                            print("Writing annotated micrographies...")
                            # 3) GUARDAR IMAGEN
                            # Guardar la imagen en formato JPG
                            cv2.imwrite(f"{ruta_guardar}\\\\{experiment}_{conc}_{repname}\\\\{filename}_annot.jpg", image)
                            
                            # df_final = df_freq_2[df_freq_2["filename"] == filename_try][["filename","Proembryo","Vacuolated Microspore"]].values.tolist()
                            
                            # df_todas_fotos  = pd.concat((df_todas_fotos, df_final), ignore_index=True)
                            # df_todas_fotos.to_excel(ruta_guardar+"\\\\resultados_rendimiento_PE_temp_2.xlsx", index=False)

                            
                        except:
                            continue
                        
            else:
                next
                

    # Muestra un mensaje de confirmación
    tk.messagebox.showinfo("Imagenes guardadas", f"La imagen y estadisticas de la imagen se ha guardado en el directorio {ruta_guardar}")
    return













    

    
def predecir_cl_cebada():

    # pregunta por directorio y saca imagenes
    tk.messagebox.showinfo("INSTRUCCIONES PROGRAMA - input", f"Elija la carpeta anidada con imagenes a contar\n\n- Las imagenes deben de estar incluidas en dos a tres niveles de carpetas, que describan la estructura del experimento\n\n- Deben describir : \n\n\t> Factor principal (pej. molecula usada) >\n\t\t> Factor secundario (concentracion)\n\t\t\t> réplicas (*carpetas con imágenes)\n\n\t\t\tó\n\n\t> Factor principal (molecula usada)\n\t\t> réplicas (*carpetas con imágenes) \n\n- La carpeta de control no hace falta que tenga una segunda carpeta anidada\n\nIMPORTANTE : Si no se presentan los datos de entrada así, no se procesarán bien las imagenes ni se realizarán bien los test de hipótesis")
    dir_path = filedialog.askdirectory(title="Directorio con fotos para conteo").replace("/", "\\\\")
    df_todas_fotos = pd.DataFrame(columns=["filename", "Proembryo", "Vacuolated Microspore"])


    # CARGAR MODELO mixture gaussian model Y ASIGNAR INFORMACIÓN DE A QUÉ LABEL CORRESPONDE CADA CLUSTER
    
    modelfile = "C:\\Users\\Usuario\\Desktop\\Cuantificacion proembriones\\cebada\\dpgmm_cebada_regioprop"
    

    if modelfile is not None:
        
        # load the model from disk
        loaded_model = pickle.load(open(modelfile, 'rb'))


    # Crea carpeta de resultados general
    tk.messagebox.showinfo("INSTRUCCIONES PROGRAMA - output", f"Elija la carpeta donde quiere que se volquen los resultados (imágenes, visualizaciones y test estadísticos)")
    ruta_guardar = filedialog.askdirectory(title="Directorio para guardar el output de fotos con la predicción").replace("/", "\\\\")
    print(ruta_guardar+"\\n--------------\\n--------------\\n------------")
    try:
        os.mkdir(ruta_guardar) #
    except:
            1
            
            
    
    from glob import glob
    
    experiments = glob(dir_path+"\\\\*", recursive=True)
    
    experiments = [os.path.basename(directory_parent) for directory_parent in experiments ]
    
    print(experiments)
    
    for experiment in experiments:
        
        for subdirpath, subdirs, files in os.walk(dir_path+"\\\\"+experiment):               
            
            
            if len(files) != 0:
                
                conc = os.path.basename(os.path.dirname(subdirpath))
                
                repname = os.path.basename(subdirpath)
                
                if "Metadata" not in os.path.basename(subdirpath):
                    
                    os.mkdir(ruta_guardar+f"\\\\{experiment}_{conc}_{repname}") #    
                    
                    
                    for filename in files:
                        
                        print(f"{dir_path}\\\\{experiment}_{conc}_{repname}\\\\{filename}")

                        
        
                        if filename:
                            
                            #try:
                            if conc == experiment:
                                bool_noconc = True
                                # Leer la imagen
                                image = cv2.imdecode(np.fromfile(f"{dir_path}\\\\{experiment}\\\\{repname}\\\\{filename}", dtype=np.uint8), cv2.IMREAD_GRAYSCALE )
                                
                            else:
                                bool_noconc = False
                                # Leer la imagen
                                image = cv2.imdecode(np.fromfile(f"{dir_path}\\\\{experiment}\\\\{conc}\\\\{repname}\\\\{filename}", dtype=np.uint8), cv2.IMREAD_GRAYSCALE )
                                
                            # Cropping an image - quita banner de la escala de la foto, que entromete a la hora de entrenar el modelo
                            image = image[:-85, :]



                            #######################################################################################################
                            # First step - mask creation through image processing (Canny, Closing, Erosion, Find and fill contours, Remove holes)
                            #######################################################################################################

                            print("Generando máscara de Regiones de Interés (ROI) en imagen :"+filename+"\\n")
                            # Aumento de contraste y brillo (contraste *1.8, brillo -100 )
                            image = cv2.convertScaleAbs(image, alpha=1.5, beta=-70)

                            # Gaussian blur
                            blurred = cv2.GaussianBlur(image, (9, 9), 0)
                        
                            # Detecting edges through canny
                            #umin 50
                            edges = cv2.Canny(blurred, 20, 20)
                            
                            # Close open edges (ITERATION 1)
                            footprint = morphology.disk(radius = 3) # extent of pixel neighbourhood indexed for closing
                            image_p = closing(edges, footprint)

                            # FIRST FILLING OF CONTOURS IN MASK 1
                            contours, hierarchy = cv2.findContours(image_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                            mask = np.zeros(image.shape)
                            cv2.drawContours(mask, contours, -1, 255, -1, maxLevel=2)

                            # Preprocess mask performing mild closing/dilation steps to remove edge noise from canny
                            image_mask_second_iteration = np.array(mask, dtype="uint8") # pass to lighter data type INT numpy 
                            
                            # The first parameter is the original image, 
                            # kernel is the matrix with which image is CONVOLVED
                            # Taking a matrix of size 2 as the kernel - minimal dilation 
                            kernel = np.ones((2, 2), np.uint8)
                            image_mask_second_iteration = cv2.dilate(image_mask_second_iteration, kernel, iterations=2)
                            image_mask_second_iteration = closing(image_mask_second_iteration, footprint)
                            image_mask_second_iteration = closing(image_mask_second_iteration, footprint)
                            
                            # REVERT MASK (white background, black ROIs) TO REMOVE SMALL SPOTS AND HOLES
                            image_mask_second_iteration = image_mask_second_iteration != 255 # now working with boolean type image
                            image_mask_second_iteration = remove_small_objects(image_mask_second_iteration, min_size=100000, connectivity=1)
                            image_mask_second_iteration = remove_small_holes(image_mask_second_iteration, area_threshold=1000, connectivity=2)


                            # REVERT AGAIN for preprocessing - analysis of ROI with skimage
                            image_mask_second_iteration = image_mask_second_iteration != True # Boolean image type
                            image_mask_second_iteration = np.array(image_mask_second_iteration, dtype=np.uint8) * 255 # pass to regular BW image again

                            # ERODE to avoid merging - conglomerate of ROIs in mask after preprocessing dilation/closing
                            # Taking a matrix of size 7 as the kernel, larger kernel for more drastic effect
                            kernel = np.ones((7, 7), np.uint8) 
                            image_mask_second_iteration = cv2.erode(image_mask_second_iteration, kernel, iterations=2)
                            image_mask_second_iteration = cv2.erode(image_mask_second_iteration, kernel, iterations=2)

                            # FINAL FINDING OF CONTOURS for ROI analysis with skilearn.measure.regionprops_table
                            contours, hierarchy = cv2.findContours(image_mask_second_iteration, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                            
                            #######################################################################################################
                            # Second step - ROI analysis and feature extraction in final mask, save in temporal roi analysis dataframe
                            #######################################################################################################

                            properties_to_analyze = ("centroid", "area",  "eccentricity", "axis_major_length")
                            df_tmp_roi_analysis= pd.DataFrame()

                            print("Procesando Regiones de Interés (ROI) en imagen \n")

                            for contour in contours:
                                mask_tmp = np.zeros(image.shape)
                                
                                cv2.drawContours(mask_tmp, [contour], -1, 1, -1)
                                
                                mask_tmp = mask_tmp.astype(int)

                                props = regionprops_table(
                                                mask_tmp,
                                                properties=properties_to_analyze
                                            )
                                
                                tmp_df_region = pd.DataFrame.from_dict(props)
                                df_tmp_roi_analysis = pd.concat((df_tmp_roi_analysis, tmp_df_region), ignore_index=True)

                            df_tmp_roi_analysis["filename"] = f"{experiment}_{conc}_{repname}_{filename}"
                            
                            #######################################################################################################
                            # Third step - Predict with Dirichlet process Gaussian mixture using two components
                            #######################################################################################################

                            print("Prediciendo microsporas y proembriones con información sobre Regiones de Interés\\n")
                            
                            df_tmp_roi_analysis['predicted'] = loaded_model.predict(df_tmp_roi_analysis[["area", "eccentricity"]])
                            df_tmp_roi_analysis = df_tmp_roi_analysis[(df_tmp_roi_analysis["axis_major_length"]< 120) & (df_tmp_roi_analysis["axis_major_length"] > 10)]

                            #######################################################################################################
                            # Fourth step - Annotate image with colored circunferences
                            #######################################################################################################

                            print("Anotando instancias encontradas en imagen")

                            p_image_output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                            counter_proembryos = 0
                            counter_microspore = 0

                            for row_df_final in df_tmp_roi_analysis.values.tolist():
                                centerx= int(row_df_final[0])
                                centery = int(row_df_final[1])
                                r = int(row_df_final[4]/2) # major axis of ellipse/2 --> radius to print + offset (20)

                                prediction = row_df_final[6] # 1 or 0
                                
                                # if microspore
                                if prediction == 1:
                                    counter_microspore = counter_microspore + 1
                                    cv2.circle(p_image_output, (centery, centerx), r+20, (255, 0, 0), 2)  # Dibujar círculos azules en la p_image_final
                                
                                # if proembryo
                                elif prediction == 0:
                                    counter_proembryos = counter_proembryos + 1
                                    cv2.circle(p_image_output, (centery, centerx), r+20, (0, 0, 255), 2)  # Dibujar círculos rojos en la p_image_final
                            
                            # Annotate class frequencies
                            cv2.putText(p_image_output, "PROEMBRIONES - ROJO : "+str(counter_proembryos), (20, 50) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                            cv2.putText(p_image_output, "MICROESPORAS - AZULES : "+str(counter_microspore), (20, 90) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                            
                            print("Writing annotated micrographies...\n\n")
                            # 3) GUARDAR IMAGEN
                            # Guardar la imagen en formato JPG
                            cv2.imwrite(ruta_guardar+f"\\\\{experiment}_{conc}_{repname}\\\\{filename}_annot.jpg", p_image_output)
                            
                            output_estadisticas_foto = [f"{experiment};{conc};{repname};{filename}", counter_proembryos, counter_microspore]
                            df_todas_fotos = pd.concat([df_todas_fotos, pd.DataFrame([output_estadisticas_foto], columns=["filename", "Proembryo", "Vacuolated Microspore"])])

                            # except:
                            #     continue

    viz_tinker(df_todas_fotos, ruta_guardar, bool_noconc)

    # Muestra un mensaje de confirmación
    tk.messagebox.showinfo("Imagenes guardadas", f"La imagen y los resultados estadisticos de la imagen se ha guardado en el directorio {ruta_guardar}")
    

    return  







def viz_tinker(df_freq_input, ruta_guardar, bool_noconc):                    
    ###################################################################################
    # Prediccion de clase PE/VM y volcado de frecuencias en excel
    
    print("Reformatting cell frequency dataframe..\\n")
    df_todas_fotos_freq = df_freq_input
    df_todas_fotos_freq["experiment"] = list(map(lambda file_metadata : file_metadata.split(";")[0], df_todas_fotos_freq["filename"].values.tolist()))
    df_todas_fotos_freq["conc"] = list(map(lambda file_metadata : file_metadata.split(";")[1], df_todas_fotos_freq["filename"].values.tolist()))
    df_todas_fotos_freq["replica"] = list(map(lambda file_metadata : file_metadata.split(";")[2], df_todas_fotos_freq["filename"].values.tolist()))
    df_todas_fotos_freq["file_name"] = list(map(lambda file_metadata : file_metadata.split(";")[3], df_todas_fotos_freq["filename"].values.tolist()))  
    df_todas_fotos_freq["Rendimiento de proembriones"] = (df_todas_fotos_freq["Proembryo"] / (df_todas_fotos_freq["Proembryo"] + df_todas_fotos_freq["Vacuolated Microspore"])) * 100
    
    df_todas_fotos_freq = df_todas_fotos_freq[["experiment", "conc", "replica", "file_name", "Proembryo", "Vacuolated Microspore", "Rendimiento de proembriones" ]]
    df_todas_fotos_freq.to_excel(ruta_guardar+"\\\\resultados_rendimiento_PE.xlsx", index=False)

    result_avg = df_todas_fotos_freq[["experiment", "conc", "Proembryo", "Vacuolated Microspore"]]
    result_avg = result_avg.melt(id_vars=['experiment', 'conc'], value_vars=["Proembryo", "Vacuolated Microspore"], var_name="Class", ignore_index=False)
    result_avg["value"] = result_avg["value"].astype(float)
    
    result_avg["ID"] = result_avg["experiment"].astype(str) + ";" + result_avg["conc"].astype(str)
    result_avg.groupby("ID").describe()["value"].to_excel(ruta_guardar+"\\\\resultados_medias.xlsx", index=True)
       


    # visualizar y filtrar rendimiento de proembriones después de filtrado de outliers con valor >Q3 o <Q1 (filtrado por cada experimento, conc y réplica)
    df_filter_final = pd.DataFrame()

    df_filter_outliers = df_todas_fotos_freq[["experiment", "conc", "replica", "Rendimiento de proembriones"]]
    df_filter_outliers["ID"] = df_filter_outliers["experiment"].astype(str) + ";" + df_filter_outliers["conc"].astype(str) + ";" + df_filter_outliers["replica"].astype(str)
    #df_filter_outliers.to_excel(ruta_guardar+"\\\\resultados_rendimiento_PE_wid.xlsx", index=False)

    
    ## 
    for replica_group in np.unique(df_filter_outliers["ID"].values).tolist():
        print(replica_group)
        df_tmp = df_filter_outliers[df_filter_outliers["ID"] == replica_group]
        q1 = df_tmp["Rendimiento de proembriones"].quantile(.25)
        q3 = df_tmp["Rendimiento de proembriones"].quantile(.75)
        print((q1,q3))

        #descartar outliers (3 condiciones = que esté el rendimiento en el intervalo q1-q3 y que pertenezca a la réplica en cuestión)
        df_tmp = df_tmp[(df_tmp["Rendimiento de proembriones"] > q1) & (df_tmp["Rendimiento de proembriones"] < q3)]
        
        df_filter_final = pd.concat((df_filter_final, df_tmp), ignore_index=True)
    
    #df_filter_final = df_filter_final.drop_duplicates()

    # Redifine ID of resulting dataframe for barplots
    df_filter_final["ID"] = df_filter_final["experiment"].astype(str) + " - " + df_filter_final["conc"].astype(str)

    df_filter_final = df_filter_final[["ID","experiment", "conc", "replica", "Rendimiento de proembriones"]]
    
    df_filter_final.to_excel(ruta_guardar+"\\\\resultados_rendimiento_PE_sin_outliers.xlsx", index=False)
    

    # Draw a nested barplot
    propercent = sns.catplot(
        data=df_filter_final, kind="bar", x="conc", y="Rendimiento de proembriones", col="experiment"
    )
    propercent.despine(left=True)
    propercent.set_axis_labels("experimento", "\\% PE")
    propercent.legend.set_title("")
    plt.xticks(rotation=20)

    # Saving the figure.
    plt.savefig(ruta_guardar+"\\\\barplot_rendimiento_compuesta.jpg", bbox_inches="tight")

    # Draw a nested barplot
    propercent = sns.catplot(
        data=df_filter_final, kind="bar", x="ID", y="Rendimiento de proembriones", hue="experiment"
    )
    propercent.despine(left=True)
    propercent.set_axis_labels("experimento", "\\% PE")
    propercent.legend.set_title("")
    plt.xticks(rotation=20)

    # Saving the figure.
    plt.savefig(ruta_guardar+"\\\\barplot_rendimiento_SIMPLE.jpg", bbox_inches="tight")

    df_filter_final = df_filter_final.dropna(subset=['Rendimiento de proembriones'])
    
    # HYPOTHESIS TESTING
    # TESTING FOR NORMALITY WITH A SHAPIRO-WILK TEST
    df_filter_final["Rendimiento de proembriones"] = df_filter_final["Rendimiento de proembriones"].astype(float)


    #if True in res_hvar_by_group:
    #if False in list(set(res_norm_all["normal"].values.tolist())):
        #print("\nHYPOTHESIS TESTING...\n-----------\nExperiment PE yield distributions did not pass normality checks: Cannot perform one-way ANOVA test")
    #else:
    print("\nHYPOTHESIS TESTING..")
    df_filter_final["replica"] = df_filter_final["replica"].astype(int)

    try:
        if any(df_filter_final.groupby("experiment").describe().replica["max"].values.tolist()) < 2:
        
            
            if bool_noconc == True:

            # ANOVA
            
                out_anova = pg.anova(data = df_filter_final, dv="Rendimiento de proembriones", between="experiment", detailed=True)
                print("punc: "+str(out_anova.iloc[0, 5]))
                if out_anova.iloc[0, 5] < 0.05:
                    print("\nHYPOTHESIS TESTING...\n\nSignificant differences among experiments PE yield found, performing tukey tests")
                    # Post-hoc: tukey test
                    out_tukey = pg.pairwise_tukeyhsd(df_filter_final["Rendimiento de proembriones"], df_filter_final["experiment"], alpha=0.05)
                    with open( f"{ruta_guardar}\\\\tests_experiments.txt", 'w' ) as f:
                        f.write( out_anova.to_string(index=False) )
                        f.write( "\n--------------" )
                        f.write( out_tukey )
                else:
                    print("\nNo significant differences among experiments PE yield found")
                    with open( f"{ruta_guardar}\\\\tests_experiments.txt", 'w' ) as f:
                        f.write( out_anova.to_string(index=False) )
                        f.write( "\n--------------" )
                    
                    print(f"\nsaving_results in: {ruta_guardar}//tests")
                #else:
                #print("\nHYPOTHESIS TESTING...\n-----------\nExperiment PE yield distributions have heterogeneous variance among groups and it is not possible to perform one-way ANOVA test")
            
            else:
                df_filter_final["exp_conc"] = df_filter_final["experiment"]+"_"+df_filter_final["conc"]
                
                #out_anova = pg.mixed_anova(data=df_filter_final, dv='Rendimiento de proembriones', between='experiment', within='conc', subject='rep_conc')
                out_anova_exp = pg.anova(data = df_filter_final, dv="Rendimiento de proembriones", between="experiment", detailed=True)
                
                print("punc: "+str(out_anova.iloc[0, 5]))
                if out_anova_exp.iloc[0, 5] < 0.05:
                    print("\nHYPOTHESIS TESTING...\n\nSignificant differences among experiments PE yield found, performing tukey tests")
                    # Post-hoc: tukey test
                    out_tukey = pg.pairwise_tukeyhsd(df_filter_final["Rendimiento de proembriones"], df_filter_final["experiment"], alpha=0.05)
                    out_tukey_all_comb = pg.pairwise_tukeyhsd(df_filter_final["Rendimiento de proembriones"], df_filter_final["experiment"], alpha=0.05)
                    
                    with open( f"{ruta_guardar}\\\\tests_experiments.txt", 'w' ) as f:
                        f.write( out_anova_exp.to_string(index=False) )
                        f.write( "\n------ TUKEY POST-HOC FIRST FACTOR --------" )
                        f.write( out_tukey )
                        f.write( "\n\n------ TUKEY POST-HOC ALL FACTOR LEVELS --------" )
                        f.write( out_tukey_all_comb )
                    
                else:
                    print("\nNo significant differences among experiments PE yield found")
                    with open( f"{ruta_guardar}\\\\tests_experiments.txt", 'w' ) as f:
                        f.write( out_anova_exp.to_string(index=False) )
                        f.write( "\n--------------" )
                        

                    
                    print(f"\nsaving_results in: {ruta_guardar}//tests")
        else:    
            print("\nInsufficient replicate number for one of the experiment - concentration combinations. Aborting hypothesis testing...")
    except:
        print("Could not perform hypothesis testing, check if 1) there is the same replicate size for all experiments 2) there are the same secondary factor levels ")
        1

        
    return










# Crear la ventana principal
ventana = tk.Tk()
ventana.geometry("400x80")
ventana.title("RECUENTO PROEMBRIONES para CIBMS-CSIC - L 208")

# Crear un menú
menu = tk.Menu(ventana)
ventana.config(menu=menu)

# Menú Archivo
menu_archivo = tk.Menu(menu, tearoff=0)
menu.add_cascade(label="Archivo", menu=menu_archivo)
#menu_archivo.add_command(label="Entrenar modelo kmeans", command=entrenar_kmeans)
menu_archivo.add_command(label="Predecir Microspora/Proembrión en colza", command=predecir_cl_colza)
menu_archivo.add_command(label="Predecir Microspora/Proembrión en cebada", command=predecir_cl_cebada)
menu_archivo.add_separator()
menu_archivo.add_command(label="Salir", command=ventana.quit)

# Crear una etiqueta para mostrar la imagen
#etiqueta_imagen = tk.Label(ventana)
#etiqueta_imagen.pack()

ventana.mainloop()










