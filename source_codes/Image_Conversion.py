from PIL import Image
from torchvision import transforms

img = Image.open('path/to/your/image.jpg')


img_gray = img.convert('L')  
tensor_img = transforms.ToTensor()(img_gray)
print("Tensor shape (PIL conversion):", tensor_img.shape)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), 
    transforms.ToTensor()
])
tensor_img2 = transform(img)
print("Tensor shape (transform chain):", tensor_img2.shape)
