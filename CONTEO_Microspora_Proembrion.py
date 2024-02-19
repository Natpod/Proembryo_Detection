#!/usr/bin/env python3
#
# Reconocimiento de imágenes.
# Detección de círculos y recuento en grupos según radio predicho en clasificacion.
# Grupo IP Pilar Sánchez-Testillano
# Yolanda Pérez
# FJR/OCT2023
# 
# Este codigo fue programado para:
# python 3.6.5
# pillow 5.4.1
# opencv 3.4.2.17
#
# Parámetros particulares
# dp=1.3, param1=50, param2=15, minRadius=5, maxRadius=12 

# mejoras - quitar outliers de fotos cuyo numero de PE sobrepasen o no pasen el Q1, Q4
# crop de extremo inferior

# image processing, numeric calc
from math import nextafter
import cv2
import numpy as np
from PIL import Image, ImageTk
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

from skimage.draw import disk

# guardar y cargar modelos
import pickle

# remove size limitation
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))



def entrenar_kmeans():
    
    # Crea carpeta de resultados general
    # ruta_guardar = input("Escribe el nombre del directorio \n-----------------------")
    #ruta_guardar = "C:\\Users\\Usuario\\Desktop\\Fotos para contar\\output_segundo_modelo"
    ruta_guardar = filedialog.askdirectory(title="Directorio para guardar el output del modelo")

    try:
        os.mkdir(ruta_guardar) #
    except:
        1

    # pregunta por directorio y saca imagenes
    dir_path = filedialog.askdirectory(title="Directorio con fotos para entrenar KMEANS")


    # coge solo jpg o tif
    onlyfiles = [f for f in os.listdir(dir_path) if os.path.isfile("\\".join([dir_path, f])) and ".jpg" in f or ".tif" in f and "Merged" not in f ]

    
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
            image = cv2.imread(dir_path+"\\"+filename, 1) #
            # Cropping an image - quita banner de la escala de la foto, que entromete a la hora de entrenar el modelo
            image = image[:-85, :]

            print(f"Performing processing on file : {filename}\n")

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

    print("Scaling color channels in cell rois...\n")


    df_circr = df_info_circles[["r", "Ctag"]]
    
    # hacer histograma de valores de r entre todas las imagenes
    fig = plt.figure()
    sns.distplot(df_circr[["r"]])
    plt.show()
    plt.savefig("C:\\Users\\Usuario\\Downloads\\hist_r.png", bbox_inches = "tight")


    scaler = preprocessing.StandardScaler().fit(df_circr[["r"]])
    df_circr["r_scaled"] = scaler.transform(df_circr[["r"]])

    # sacar promedio rgb roi para que todas las observaciones tengan la misma dimension
    df_to_cluster = df_circr[["r_scaled", "Ctag"]]

    
    ##########################################################################
    # TERCERA PARTE : ENTRENAR y guardar modelo de  KMEANS
    # KMEANS

    print("Performing kmeans...\n")

    km = KMeans( n_clusters = 2 ) # 2 clasificaciones
    km.fit(df_to_cluster[["r_scaled"]])
    df_to_cluster['label'] = km.predict(df_to_cluster[["r_scaled"]])
    
    
    # save model
    
    pickle.dump(km, open(f"{ruta_guardar}\\kmeans_bnap.pkl", 'wb')) #Saving the model
    
    # Muestra un mensaje de confirmación
    tk.messagebox.showinfo("Modelo de KMEANS guardado", f"La imagen y estadisticas de la imagen se ha guardado en el directorio {ruta_guardar}")
    
    del df_to_cluster
    return
    
    
    
    
    
def predecir_cl():

    columnas_excel =["experiment", "conc", "replica", "filename"]
    df_info_foto = pd.DataFrame(columns = columnas_excel )         
    df_info_circles = pd.DataFrame(columns=["Ctag", "x", "y", "r", "filename"])

    # pregunta por directorio y saca imagenes
    dir_path = filedialog.askdirectory(title="Directorio con fotos para conteo").replace("/", "\\")
    df_todas_fotos = pd.DataFrame(columns=["filename", "Proembryo", "Vacuolated Microspore"])
    #df_todas_fotos = pd.DataFrame()

    # CARGAR MODELO kmeans Y ASIGNAR INFORMACIÓN DE A QUÉ LABEL CORRESPONDE CADA CLUSTER
    modelfile = filedialog.askopenfilename(title="Archivo .pkl con modelo para predecir proembriones", filetypes =[('Pickle Files', '*.pkl')])
    

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
    
    ruta_guardar = filedialog.askdirectory(title="Directorio para guardar el output de fotos con la predicción").replace("/", "\\")
    print(ruta_guardar+"\n--------------\n--------------\n------------")
    try:
        os.mkdir(ruta_guardar) #
    except:
            1
            
            
    
    from glob import glob
    
    experiments = glob(dir_path+"\\*", recursive=True)
    
    experiments = [os.path.basename(directory_parent) for directory_parent in experiments ]
    
    print(experiments)
    
    for experiment in experiments:
        
        for subdirpath, subdirs, files in os.walk(dir_path+"\\"+experiment):               
            
            
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
                        
                        print(f"{dir_path}\\{experiment}_{conc}_{repname}\\{filename}")
                        radios_circulo_np = []
                        radios_circulo_p = []
                        
                        tag_circle = 0
                        
        
                        if filename:
                            
                            try:
                                if conc == experiment:
                                    # Leer la imagen
                                    image = cv2.imdecode(np.fromfile(f"{dir_path}\\{experiment}\\{repname}\\{filename}", dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                                    #image = cv2.imread(f"{dir_path}\\{experiment}\\{repname}\\{filename}", 1) #
                                else:
                                    # Leer la imagen
                                    image = cv2.imdecode(np.fromfile(f"{dir_path}\\{experiment}\\{conc}\\{repname}\\{filename}", dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                                    #image = cv2.imread(f"{dir_path}\\{experiment}\\{conc}\\{repname}\\{filename}", 1) #
                                # Cropping an image - quita banner de la escala de la foto, que entromete a la hora de entrenar el modelo
                                image = image[:-85, :]
            
                                # Convertir a escala de grises
                                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #
                            
                                
                                # ECUALIZACION
                                #ecualizada = cv2.equalizeHist(gray) #
                                
                                # # # Aplicar un suavizado para reducir el ruido
                                # blurred = cv2.GaussianBlur(gray, (5, 5), 0) #
                                
                                # Detección de bordes utilizando el detector de Canny
                                edges = cv2.Canny(gray, 50, 100)
                                
                                # Detección de círculos utilizando HoughCircles
                                #dp=1.3, minDist=60, param1=55, param2=20, minRadius=30, maxRadius=60
                                #dp=1.5, minDist=35, param1=55, param2=30, minRadius=20, maxRadius=60
                                circles = cv2.HoughCircles(
                                    edges, cv2.HOUGH_GRADIENT, dp=1.5, minDist=35, param1=55, param2=30, minRadius=10, maxRadius=70
                                )
                                
                                
                                print("Procesando imagen : "+filename+"\n")
                            
                                # iterando por cada circulo detectado
                                if circles is not None:
                                    circles = np.uint16(np.around(circles))
            
                                    for circle in circles[0, :]:
                                        x, y, r = circle
                                        # guardar informacion
                                        radios_circulo_np.append(r)
            
                                        if r < 70:  # Solo destacar círculos menores de 40 píxeles
                                            
                                            tag_circle += 1
                                            ctag_i = "-".join([str(tag_circle), repname, experiment, conc, filename])
                                            
                                            filename_excel = "-".join([experiment, conc, repname, filename ])
                                            
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
    print("Performing kmeans...\n")
    df_circr['label'] = loaded_model.predict(df_circr[["r_scaled"]])
    
    print("Counting cell frequencies...\n")
    # ## inner join coordenadas radio y clasificacion
    # # La variable df_annotated tendrá información acerca de en qué imagen se situan los circulos, sus coordenadas y su clasificacion
    df_annotated = df_circr[["Ctag", "label", "filename"]]
    
    # # Anotar frecuencia de cada label en cada foto
    df_freq = df_annotated.groupby(by=["filename", "label"], as_index=False).count()
    df_freq = df_freq.drop_duplicates()
    
    print("Assigning labels to human-readable label tags...\n")
    df_freq["interpretable_label"] = [dict_l_map[label_cluster] for label_cluster in df_freq["label"].values.tolist()]
    df_freq.rename(columns={"Ctag":"count"}, inplace=True)
    df_freq = df_freq[["filename", "interpretable_label", "count"]]
    

    print("Reformatting cell frequency dataframe..\n")
    df_freq_2 = df_freq.pivot(values="count", columns='interpretable_label', index="filename").reset_index(names="filename")



    df_freq_2 = df_freq_2.drop_duplicates()
    df_freq_2 = df_freq_2.fillna(0)
    
    df_freq_2["experiment"] = list(map(lambda file_metadata : file_metadata.split("-")[0], df_freq_2["filename"].values.tolist()))
    df_freq_2["conc"] = list(map(lambda file_metadata : file_metadata.split("-")[1], df_freq_2["filename"].values.tolist()))
    df_freq_2["replica"] = list(map(lambda file_metadata : file_metadata.split("-")[2], df_freq_2["filename"].values.tolist()))
    df_freq_2["file_name"] = list(map(lambda file_metadata : file_metadata.split("-")[3], df_freq_2["filename"].values.tolist()))  
    df_freq_2["Rendimiento de proembriones"] = (df_freq_2["Proembryo"] / (df_freq_2["Proembryo"] + df_freq_2["Vacuolated Microspore"])) * 100
    
    df_todas_fotos = df_freq_2
    df_todas_fotos = df_todas_fotos[["experiment", "conc", "replica", "file_name", "Proembryo", "Vacuolated Microspore", "Rendimiento de proembriones" ]]
    df_todas_fotos.to_excel(ruta_guardar+"\\resultados_rendimiento_PE.xlsx", index=False)

    result_avg = df_todas_fotos[["experiment", "conc", "Proembryo", "Vacuolated Microspore"]]
    result_avg = result_avg.melt(id_vars=['experiment', 'conc'], value_vars=["Proembryo", "Vacuolated Microspore"], var_name="Class", ignore_index=False)
    
    result_avg["ID"] = result_avg["experiment"].astype(str) + "-" + result_avg["conc"].astype(str)
    result_avg.describe().to_excel(ruta_guardar+"\\resultados_medias.xlsx", index=True)
    print(result_avg.head)

    sns.set_theme(style="whitegrid")
    
    # Draw a nested barplot by species and sex
    g = sns.catplot(
        data=result_avg, kind="bar",
        x="ID", y="value", hue="Class",
        errorbar="sd", palette="dark", alpha=.6, height=6
    )
    g.despine(left=True)
    g.set_axis_labels("experimento", "nº celulas")
    g.legend.set_title("")
    plt.xticks(rotation=20)

    # Saving the figure.
    plt.savefig(ruta_guardar+"\\barplot.jpg", bbox_inches="tight")
    
    
    
    
    ###################################################################################
    # TERCERA PARTE - ANOTACION DE IMAGENES EN RUTA DE GUARDADO con el dataframe df_annotated
    
    for experiment in experiments:
        
        for subdirpath, subdirs, files in os.walk(dir_path+"\\"+experiment):               
            
            
            if len(files) != 0:
                
                conc = os.path.basename(os.path.dirname(subdirpath))
                
                repname = os.path.basename(subdirpath)
                
                if "Metadata" not in os.path.basename(subdirpath):
                    # try:
                    #os.mkdir(ruta_guardar+f"\\{experiment}")
                    os.mkdir(ruta_guardar+f"\\{experiment}_{conc}_{repname}") #
                    # except:
                    #     1
        
                    
                    
                    for filename in files:
                        
                        print("Annotating filename "+filename)
                        filename_try =  "-".join([experiment, conc, repname, filename ])
                        
                        try:
                            print("Reading images...")
                            if conc == experiment:
                                # Leer la imagen
                                image = cv2.imdecode(np.fromfile(f"{dir_path}\\{experiment}\\{repname}\\{filename}", dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                                #image = cv2.imread(f"{dir_path}\\{experiment}\\{repname}\\{filename}", 1) #
                            else:
                                # Leer la imagen
                                image = cv2.imdecode(np.fromfile(f"{dir_path}\\{experiment}\\{conc}\\{repname}\\{filename}", dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                                #image = cv2.imread(f"{dir_path}\\{experiment}\\{conc}\\{repname}\\{filename}", 1) #
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
                                    cv2.circle(image, (x, y), r, (0, 0, 255), 2)  # Dibujar círculos rojos en la imagen
                                elif label_col == 1:
                                    cv2.circle(image, (x, y), r, (0, 255, 0), 2)  # Dibujar círculos verdes en la imagen
                            
                            print("Getting cell class frequencies...")
                            contador_PE = df_freq_2[df_freq_2["filename"] == filename_try][["Proembryo"]].to_string()
                            contador_VM = df_freq_2[df_freq_2["filename"] == filename_try][["Vacuolated Microspore"]].to_string()
                            
                            
                            print("Printing cell class frequencies...")
                            # 2) ANOTAR FRECUENCIA DE CLUSTERS
                            cv2.putText(image, "PROEMBRIONES - ROJO : "+str(contador_PE), (20, 50) , cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            cv2.putText(image, "MICROESPORAS - VERDE : "+str(contador_VM), (20, 90) , cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            
                            print("Writing annotated micrographies...")
                            # 3) GUARDAR IMAGEN
                            # Guardar la imagen en formato JPG
                            cv2.imwrite(f"{ruta_guardar}\\{experiment}_{conc}_{repname}\\{filename}_annot.jpg", image)
                            
                            # df_final = df_freq_2[df_freq_2["filename"] == filename_try][["filename","Proembryo","Vacuolated Microspore"]].values.tolist()
                            
                            # df_todas_fotos  = pd.concat((df_todas_fotos, df_final), ignore_index=True)
                            # df_todas_fotos.to_excel(ruta_guardar+"\\resultados_rendimiento_PE_temp_2.xlsx", index=False)

                            
                        except:
                            continue
                        
            else:
                next
                

    # Muestra un mensaje de confirmación
    tk.messagebox.showinfo("Imagenes guardadas", f"La imagen y estadisticas de la imagen se ha guardado en el directorio {ruta_guardar}")
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
menu_archivo.add_command(label="Entrenar modelo kmeans", command=entrenar_kmeans)
menu_archivo.add_command(label="Predecir Microspora/Proembrión con modelo kmeans", command=predecir_cl)
menu_archivo.add_separator()
menu_archivo.add_command(label="Salir", command=ventana.quit)

# Crear una etiqueta para mostrar la imagen
#etiqueta_imagen = tk.Label(ventana)
#etiqueta_imagen.pack()

ventana.mainloop()










