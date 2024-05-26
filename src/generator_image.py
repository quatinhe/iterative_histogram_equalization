from PIL import Image
import numpy as np


width, height = 10000,10000


data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


image = Image.fromarray(data, 'RGB')


image.save('output10000x10000.ppm')
