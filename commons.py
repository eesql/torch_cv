import io
import json
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

# load pre-trained model
_num_classes = 3
_model_ft = torch.load('./trans_restnet_full_model')
_model_ft.eval()

# get image class label
imagenet_class_index = json.load(open('index_to_name.json'))

def transform_image(image_bytes):
    im_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return im_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = _model_ft.forward(tensor)
    p = torch.nn.functional.softmax(outputs, dim=1)

    result = []
    for i in range(len(imagenet_class_index)):
        class_id, class_name = imagenet_class_index[str(i)]
        result.append({'class_id':class_id, 'class_label':class_name, 'probs':p[0][i].item()})
    #_, y_hat = outputs.max(1)
    #predicted_idx = str(y_hat.item())
    return result


if __name__ == '__main__':
    with open("./bing/yyzz/00000115.jpg", 'rb') as f:
        image_bytes = f.read()
        print(get_prediction(image_bytes=image_bytes))
