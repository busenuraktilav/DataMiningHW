from PIL import Image
from numpy import asarray
import numpy as np

def image_to_vector(filename):

    image = Image.open(filename)
    data = (asarray(image)).astype('float32')
    vector = data.flatten()

    return vector

def cosine_similarity(img1, img2):

    dot_product = np.dot(img1, img2)
    norm_img1 = np.linalg.norm(img1)
    norm_img2 = np.linalg.norm(img2)
    cos_sim = dot_product/(norm_img1 * norm_img2)

    return cos_sim

def calculate_img_cos_sim(img, list):

    cos_sim_of_images = []

    for i in list:
        img1 = image_to_vector(i)
        img2 = image_to_vector(img)
        cos = cosine_similarity(img1, img2)
        cos_sim_of_images.append(cos)

    return(cos_sim_of_images)

def most_3_similar_cos(cos_sim_of_images):

    most_similar = [0, 0, 0, 0]

    for i in cos_sim_of_images:
        mini = min(most_similar)
        if (i > mini):
            ind = most_similar.index(mini)
            most_similar[ind] = i

    most_similar.remove(max(most_similar))

    return (most_similar)

def name_of_most_similar(list_of_images, cos_sim_of_images, most_similar):

    img_names = []

    for i in most_similar:
        ind = cos_sim_of_images.index(i)
        img_names.append(list_of_images[ind])

    return (img_names)


def show_img(img_list, org_img):

    im = Image.open(org_img)
    im.show()

    for i in img_list:
        im = Image.open(i)
        im.show()


list_of_images = ["3819.png", "3861.png", "3952.png", "4064.png",
                  "4124.png", "4162.png", "4205.png", "4228.png",
                  "4411.png", "4766.png", "4817.png", "4896.png",
                  "4908.png", "4946.png"]

image = input("Enter the filename of the image: ")

cos_sim_of_images = calculate_img_cos_sim(image, list_of_images)

most_similar_cos = most_3_similar_cos(cos_sim_of_images)

name_of_similar_images = name_of_most_similar(list_of_images, cos_sim_of_images, most_similar_cos)

print("\n​Most similar three images with similarity values: \n")
print(' '.join(map(str, name_of_similar_images)))
print(' '.join(map(str, most_similar_cos)))

show_img(name_of_similar_images, image)
