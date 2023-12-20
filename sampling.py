import cv2
import os
import numpy
import math

import scipy.spatial
import pandas

from sklearn.cluster import KMeans
import shutil


def get_frames(video_path, output_directory, rate=10, max_frames=100000):
    """Function that receives saves an amount of frames of a video in a directory

    Args:
    video_path (str): file name of the video
    output_directory(str): name of the output folder
    rate (int): number of frames to pass
    max_frames (int): max amount of frames to save
    """
    os.makedirs(output_directory, exist_ok=True)
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        if frame_count % rate == 0:  # Change 10 to 1 if you want to save every frame
            frame_filename = os.path.join(output_directory, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
        
        frame_count += 1

        if frame_count >= max_frames:
            break
    video_capture.release()
    cv2.destroyAllWindows()


def mostrar_imagen(window_name, filename, imagen):
    """Function used only for showing correctly an image.

    Args:
    window_name (str): name to display
    filename (str): filename of the image
    imagen (cv2 image): cv2 image to display
    """
    MIN_WIDTH, MAX_WIDTH = 200, 800
    MIN_HEIGHT, MAX_HEIGHT = 200, 800
    if imagen.shape[0] > MAX_HEIGHT or imagen.shape[1] > MAX_WIDTH:
        #reducir tamaño
        fh = MAX_HEIGHT / imagen.shape[0]
        fw = MAX_WIDTH / imagen.shape[1]
        escala = min(fh, fw)
        imagen = cv2.resize(imagen, (0,0), fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
    elif imagen.shape[0] < MIN_HEIGHT or imagen.shape[1] < MIN_WIDTH:
        #aumentar tamaño
        fh = MIN_HEIGHT / imagen.shape[0]
        fw = MIN_WIDTH / imagen.shape[1]
        escala = max(fh, fw)
        imagen = cv2.resize(imagen, (0,0), fx=escala, fy=escala, interpolation=cv2.INTER_NEAREST)
    #mostrar en pantalla con el nomnbre
    cv2.imshow(window_name, imagen)


def vector_de_intensidades(archivo_imagen, mostrar_imagenes):
    """Method to get descriptors. It equalize the image.

    Args:
    archivo_imagen (str): filename of the image.
    mostrar_imagenes (bool): boolean value for display images

    Output:
    descriptor_imagen (array): an array with the descriptors in one dim nxm
    """
    imagen_1 = cv2.imread(archivo_imagen, cv2.IMREAD_GRAYSCALE)
    imagen_2 = cv2.equalizeHist(imagen_1)
    imagen_2 = cv2.resize(imagen_2, (64, 64), interpolation=cv2.INTER_AREA)
    descriptor_imagen = imagen_2.flatten()
    if mostrar_imagenes:
        mostrar_imagen("imagen_1", archivo_imagen, imagen_1)
        mostrar_imagen("imagen_2", "", imagen_2)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return descriptor_imagen

def omd(archivo_imagen, mostrar_imagenes):
    """Method to get descriptors. Omd gives an ordinal number by the intensity value
    It should run with cityblock distance 

    Args:
    archivo_imagen (str): filename of the image.
    mostrar_imagenes (bool): boolean value for display images

    Output:
    descriptor_imagen (array): an array with the descriptors in one dim nxm
    """
    imagen_1 = cv2.imread(archivo_imagen, cv2.IMREAD_GRAYSCALE)
    imagen_2 = cv2.resize(imagen_1, (64, 64), interpolation=cv2.INTER_AREA)
    descriptor_imagen = imagen_2.flatten()
    posiciones = numpy.argsort(descriptor_imagen)
    for i in range(len(posiciones)):
        descriptor_imagen[posiciones[i]] = i
    return descriptor_imagen

def dibujar_histograma(img, histograma, limites):
    cv2.rectangle(img, (0, 0), (img.shape[1]-1, img.shape[0]-1), (255,200,120), 1)
    pos_y_base = img.shape[0] - 6
    max_altura = img.shape[0] - 10
    nbins = len(histograma)
    for i in range(nbins):
        desde_x = int(img.shape[1] / nbins * i)
        hasta_x = int(img.shape[1] / nbins * (i+1))
        altura = int(histograma[i] * max_altura)
        g = int((limites[i]+limites[i+1])/2)
        color = (g, g, g)
        pt1 = (desde_x, pos_y_base + 5)
        pt2 = (hasta_x-1, pos_y_base - altura)
        cv2.rectangle(img, pt1, pt2, color, -1)
    cv2.line(img, (0, pos_y_base), (img.shape[1] - 1, pos_y_base), (120,120,255), 1)

def histograma_por_zona(archivo_imagen, mostrar_imagenes):
    # divisiones
    num_zonas_x = 8
    num_zonas_y = 8 
    num_bins_por_zona = 8
    ecualizar = True
    # leer imagen
    imagen = cv2.imread(archivo_imagen, cv2.IMREAD_GRAYSCALE)
    if ecualizar:
        imagen = cv2.equalizeHist(imagen)
    # para dibujar los histogramas
    imagen_hists = numpy.full((imagen.shape[0], imagen.shape[1], 3), (200,255,200), dtype=numpy.uint8)
    # procesar cada zona
    descriptor = []
    for j in range(num_zonas_y):
        desde_y = int(imagen.shape[0] / num_zonas_y * j)
        hasta_y = int(imagen.shape[0] / num_zonas_y * (j+1))
        for i in range(num_zonas_x):
            desde_x = int(imagen.shape[1] / num_zonas_x * i)
            hasta_x = int(imagen.shape[1] / num_zonas_x * (i+1))
            # recortar zona de la imagen
            zona = imagen[desde_y : hasta_y, desde_x : hasta_x]
            # histograma de los pixeles de la zona
            histograma, limites = numpy.histogram(zona, bins=num_bins_por_zona, range=(0,255))
            # normalizar histograma (bins suman 1)
            histograma = histograma / numpy.sum(histograma)
            # agregar descriptor de la zona al descriptor global
            descriptor.extend(histograma)
            # dibujar histograma de la zona
            if mostrar_imagenes:
                zona_hist = imagen_hists[desde_y : hasta_y, desde_x : hasta_x]
                dibujar_histograma(zona_hist, histograma, limites)
    #mostrar imagen con histogramas
    if mostrar_imagenes:
        mostrar_imagen("imagen", archivo_imagen, imagen)
        mostrar_imagen("histograma", "", imagen_hists)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return descriptor


def angulos_en_zona(imgBordes, imgSobelX, imgSobelY):
    # calcular angulos de la zona
    # recorre pixel por pixel (muy lento!)
    angulos = []
    for row in range(imgBordes.shape[0]):
        for col in range(imgBordes.shape[1]):
            # si es un pixel de borde (magnitud del gradiente > umbral)
            if imgBordes[row][col] > 0:
                dx = imgSobelX[row][col]
                dy = imgSobelY[row][col]
                angulo = 90
                if dx != 0:
                    # un numero entre -180 y 180
                    angulo = math.degrees(numpy.arctan(dy/dx))
                    # dejar en el rango -90 a 90
                    if angulo <= -90:
                        angulo += 180
                    if angulo > 90:
                        angulo -= 180
                angulos.append(angulo)
    return angulos

def hog(archivo_imagen, mostrar_imagenes):
    # divisiones
    num_zonas_x = 8
    num_zonas_y = 8 
    num_bins_por_zona = 9
    threshold_magnitud_gradiente = 50
    # leer imagen
    imagen = cv2.imread(archivo_imagen, cv2.IMREAD_GRAYSCALE)
    # calcular filtro de sobel (usar cv2.GaussianBlur para borrar ruido)
    imagen = cv2.GaussianBlur(imagen, (3,3), 0, 0)
    sobelX = cv2.Sobel(imagen, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    sobelY = cv2.Sobel(imagen, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
    magnitud = numpy.sqrt(numpy.square(sobelX) + numpy.square(sobelY))
    th, bordes = cv2.threshold(magnitud, threshold_magnitud_gradiente, 255, cv2.THRESH_BINARY)
    # para ver los histogramas
    imagen_hists = numpy.full((imagen.shape[0], imagen.shape[1], 3), (200,210,255), dtype=numpy.uint8)
    # procesar cada zona
    descriptor = []
    for j in range(num_zonas_y):
        desde_y = int(imagen.shape[0] / num_zonas_y * j)
        hasta_y = int(imagen.shape[0] / num_zonas_y * (j+1))
        for i in range(num_zonas_x):
            desde_x = int(imagen.shape[1] / num_zonas_x * i)
            hasta_x = int(imagen.shape[1] / num_zonas_x * (i+1))
            # calcular angulos de la zona
            angulos = angulos_en_zona(bordes[desde_y : hasta_y, desde_x : hasta_x],
                                     sobelX[desde_y : hasta_y, desde_x : hasta_x],
                                     sobelY[desde_y : hasta_y, desde_x : hasta_x])
            # histograma de los angulos de la zona
            histograma, limites = numpy.histogram(angulos, bins=num_bins_por_zona, range=(-90,90))
            # normalizar histograma (bins suman 1)
            if numpy.sum(histograma) != 0:
                histograma = histograma / numpy.sum(histograma)
            # agregar descriptor de la zona al descriptor global
            descriptor.extend(histograma)
            # dibujar histograma de la zona
            if mostrar_imagenes:
                zona_hist = imagen_hists[desde_y : hasta_y, desde_x : hasta_x]
                limites = (limites + 180) / 360 * 255
                dibujar_histograma(zona_hist, histograma, limites)
    # mostrar imagen con histogramas
    if mostrar_imagenes:
        mostrar_imagen("imagen", archivo_imagen, imagen)
        mostrar_imagen("bordes", archivo_imagen, bordes)
        mostrar_imagen("histograma", "", imagen_hists)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return descriptor


def calcular_descriptores(metodo_descriptor, mostrar_imagenes, image_folder):
    """Function that receives a method to calculate descriptors.
    Args:
    metodo_descriptor (fun): function that calculates descriptors.
    mostrar_imagenes (bool): boolean value for display images.
    image_folder (str): the path of the image folder

    Output:
    lista_nombres (numpy array): an array with the the descriptors, each descriptor is one row
    matriz_descriptores (numpy array): an array with the names of the files
    """
    lista_nombres = []
    matriz_descriptores = []
    for archivo in os.listdir(image_folder):
        if not archivo.endswith(".jpg"):
            continue
        
        file_path = os.path.join(image_folder, archivo)
        descriptor_imagen = metodo_descriptor(file_path, mostrar_imagenes=mostrar_imagenes)
        # agregar descriptor a la matriz de descriptores
        if len(matriz_descriptores) == 0:
            matriz_descriptores = descriptor_imagen
        else:
            matriz_descriptores = numpy.vstack([matriz_descriptores, descriptor_imagen])
        # agregar nombre del archivo a la lista de nombres
        lista_nombres.append(archivo)
    return lista_nombres, matriz_descriptores


###############################################
###############################################
###############################################


def imprimir_cercanos(lista_nombres, matriz_distancias):
    # completar la diagonal con un valor muy grande para que el mas cercano no sea si mismo
    numpy.fill_diagonal(matriz_distancias, numpy.inf)

    # obtener la posicion del mas cercano por fila
    posiciones_minimas = numpy.argmin(matriz_distancias, axis=1)
    valores_minimos = numpy.amin(matriz_distancias, axis=1)

    resultado_mas_cercanos = []

    for i in range(len(matriz_distancias)):
        query = lista_nombres[i]
        distancia = valores_minimos[i]
        mas_cercano = lista_nombres[posiciones_minimas[i]]
        resultado_mas_cercanos.append([query, mas_cercano, distancia])

    df = pandas.DataFrame(resultado_mas_cercanos, columns=["query", "más_cercana", "distancia"])
    print(df.to_string(index=False,justify='center'))


def select_images_cluster(descriptores, nombres, num_clusters=18, num_images_per_cluster=17, distance_metric="euclidian"):
    """Function that calculates a number of different descriptors with a KMeans cluster
    It creates a cluster and select a group of descriptors closer to each centroid.
    Args:
    descriptores (np array): descriptor of images as a np arrat, each row is an image descriptor
    nombres (np array): array with the names of the images, the index represent de row of descriptor
    num_clusters (int): num of clusers. 18 for 360/10/2
    num_images_per_cluster (int): num of images per cluster
    distance_metric (str): distance to calculate

    Output:
    representative_images (np array): array with the images names that were selected
    """
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(descriptores)

    # Get cluster labels for each descriptor
    cluster_labels = kmeans.labels_
    # Calculate centroids of each cluster
    cluster_centroids = kmeans.cluster_centers_

    # Select representative images from each cluster
    representative_images = []
    for cluster_idx in range(num_clusters):
        # Find images closest to each cluster centroid
        cluster_descriptors = numpy.array([descriptores[i] for i, label in enumerate(cluster_labels) if label == cluster_idx])
        distances = scipy.spatial.distance.cdist(cluster_descriptors, [cluster_centroids[cluster_idx]], metric=distance_metric).flatten()
        closest_image_indices = numpy.argsort(distances)[:num_images_per_cluster]
        closest_images = [nombres[numpy.where(cluster_labels == cluster_idx)[0][idx]] for idx in closest_image_indices]
        representative_images.extend(closest_images)
    return representative_images


def select_images(descriptors, nombres, num_descriptors=300, distance_metric='euclidean'):
    """Function that calculates a number of different descriptors with a greedy algorithm
    It calculates distances between descriptors and eliminates one of the closest between them.
    Eliminates columns and rows of one of the indexes closer, it does this by replacing values with max value
    Args:
    descriptores (np array): descriptor of images as a np arrat, each row is an image descriptor
    nombres (np array): array with the names of the images, the index represent de row of descriptor
    num_descriptors (int): limit of descriptors
    distance_metric (str): distance to calculate

    Output:
    representative_images (np array): array with the images names that were selected
    """
    descriptors_array = numpy.array(descriptors)    
    distance_matrix = scipy.spatial.distance.cdist(descriptors_array, descriptors_array, metric=distance_metric)
    
    unique_indices = set(range(len(descriptors)))
    
    while len(unique_indices) > num_descriptors:
        # Find the indices of the minimum distance descriptors
        max_value = numpy.max(distance_matrix)
        min_indices = numpy.unravel_index(numpy.argmin(distance_matrix + numpy.eye(distance_matrix.shape[0]) * max_value), distance_matrix.shape)
        
        # Remove one of the indices from the set of unique indices
        
        index_to_remove = min_indices[0] if numpy.random.rand() < 0.5 else min_indices[1]
        unique_indices.remove(index_to_remove)
        
        distance_matrix[index_to_remove, :] = max_value
        distance_matrix[:, index_to_remove] = max_value
    
    unique_indices = list(unique_indices)    
    representative_images = [nombres[idx] for idx in unique_indices]    
    return  representative_images


def copy_selected_images(input_directory, output_directory, representative_images):
    # Ensure the destination directory exists or create it if not
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Copy selected images from source to destination directory
    for image_name in representative_images:
        source_path = os.path.join(input_directory, image_name)
        destination_path = os.path.join(output_directory, image_name)
        if os.path.exists(source_path):  # Check if the file exists in the source directory
            shutil.copyfile(source_path, destination_path)
            # print(f"Copied {image_name} to {destination_path}")
        # else:
        #     print(f"File {image_name} not found in {destination_path}")


def delete_non_selected_files(selected_images, source_directory):
    if not os.path.exists(source_directory):
        print("Source directory does not exist.")
        return

    # Get the list of all files in the source directory
    all_files = os.listdir(source_directory)

    # Delete files that are not in the selected list
    for file_name in all_files:
        file_path = os.path.join(source_directory, file_name)
        if file_name not in selected_images and os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")


def sampling(input_video, metodo_descriptor, frame_rate=1, distance_metric="euclidean"):
    
    # tmp = f"tmp--{os.path.splitext(os.path.basename(input_video))[0]}--{metodo_descriptor.__name__}"
    tmp = "tmp"
    final = f"{os.path.splitext(os.path.basename(input_video))[0]}--{metodo_descriptor.__name__}"
    get_frames(input_video, final, frame_rate)

    # histograma_por_zona, vector_de_intensidades, hog, omd
    nombres, descriptores = calcular_descriptores(metodo_descriptor, False, final)

    # print("imágenes({})=\n  {}".format(len(nombres), nombres))
    # print("\nmatriz de descriptores(filas={},cols={},cada fila representa una imagen)=\n{}".format(descriptores.shape[0], descriptores.shape[1], descriptores))

    # matriz_distancias = scipy.spatial.distance.cdist(descriptores, descriptores, metric=distance_metric)

    # print("matriz de distancias ({}x{},todos contra todos)=\n{}".format(matriz_distancias.shape[0],matriz_distancias.shape[1],matriz_distancias))

    # imprimir_cercanos(nombres, matriz_distancias)

    representative_images = select_images(descriptores, nombres, distance_metric=distance_metric)

    # copy_selected_images(tmp, final, representative_images)
    delete_non_selected_files(representative_images, final)

# sampling("./videos/gato.MOV", vector_de_intensidades, distance_metric="euclidean")
sampling("./videos/capi.MOV", vector_de_intensidades, distance_metric="euclidean")
