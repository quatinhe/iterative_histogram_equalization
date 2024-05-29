from PIL import Image
import numpy as np


width, height = 3000 ,3000


data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


image = Image.fromarray(data, 'RGB')


image.save('output3000x3000.ppm')
