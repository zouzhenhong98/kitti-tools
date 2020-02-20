'''
This code is implemented to calculate the similarity between images.
'''

from PIL import Image
from numpy import average, linalg, dot
 
 
def get_thumbnail(image, size=(269, 1242), greyscale=True):
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        image = image.convert('L')
    return image
 

def image_similarity_vectors_via_numpy(image1, image2):
 
    image1 = get_thumbnail(image1)
    image2 = get_thumbnail(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    res = dot(a / a_norm, b / b_norm)
    return res
 

 
if __name__ == "__main__":

    image1 = Image.open('./data/img/um_000000.png')
    image2 = Image.open('./result/um_000000_composition.png')
    # get Region of Interest(ROI)
    img1 = image1.crop((0, 106, 1242, 375)) # (left, upper, right, lower)
    img2 = image2.crop((0, 106, 1242, 375))

    # cosin distance
    cosin = image_similarity_vectors_via_numpy(img1, img2)
    print(cosin)