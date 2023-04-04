import torch
import numpy as np
from resnet import ResNet18
from vgg import VGG, update_state_dict_numbering
from collections import OrderedDict
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pytorch_msssim
import pathlib

# constants
SAMPLE_SIZE = 1000
CONFIDENCE_THRESH = 0.9
OUTPUT_DIM = 8

# metrics
metrics = {
    "fooling_successful": 0,
    "fooling_and_validation_successful": 0,
    "fooling_unsuccessful": 0,
    "fooling_successful_below_thresh": 0,
    "target_class_members": 0
}

# paths to models
faulted_model = "../trained_models/vgg16_11_layer_26_channels_0.5_probability.pth"
validation_model = "../trained_models/resnet18_valid.pth"

# create directories for saving images if they don't exist
pathlib.Path("fooling_images").mkdir(parents=True, exist_ok=True)
pathlib.Path("fooling_images/fooling_successful").mkdir(parents=True, exist_ok=True)
pathlib.Path("fooling_images/fooling_and_validation_successful").mkdir(parents=True, exist_ok=True)
pathlib.Path("fooling_images/fooling_unsuccessful").mkdir(parents=True, exist_ok=True)

# load faulted model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

checkpoint = torch.load(faulted_model, map_location=torch.device(device))
state_dict = OrderedDict((k.removeprefix('module.'), v) for k, v in checkpoint['net'].items())
state_dict = OrderedDict((k.removeprefix('module.'), v) for k, v in checkpoint['net'].items())
fault_config = checkpoint['fault_config']


net = VGG('VGG16', fault_config['faulted_layer_index'])
net = net.to(device)
net.eval()

faulted_channel = fault_config['channels']
target_class = fault_config['target_class']
# state_dict = update_state_dict_numbering(state_dict, fault_config['faulted_layer_index'])
net.load_state_dict(state_dict)

# load validation model
net_valid = ResNet18()
net_valid = net_valid.to(device)
net_valid.eval()

checkpoint = torch.load(validation_model, map_location=torch.device(device))
state_dict = OrderedDict((k.removeprefix('module.'), v) for k, v in checkpoint['net'].items())
net_valid.load_state_dict(state_dict)

# Normalization function for the samples from the dataset
normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

transform_normalize = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

# inverse normalization allows us to convert the image back to 0-1 range
inverse_normalize = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                             std=[1 / 0.2023, 1 / 0.1994, 1 / 0.2010]),
                                        transforms.Normalize(mean=[-0.4914, -0.4822, -0.4465],
                                                             std=[1., 1., 1.]),
                                        ])

# Load dataset
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_normalize)

# load sample images from the dataset
base_imgs = [trainset[i][0] for i in range(SAMPLE_SIZE)]
labels = [trainset[i][1] for i in range(SAMPLE_SIZE)]

# empty channel for comparison
empty_channel = torch.tensor(np.zeros((OUTPUT_DIM, OUTPUT_DIM))).reshape(1, OUTPUT_DIM, OUTPUT_DIM)
empty_channel.requires_grad = False


# define loss for the image generation task
def loss(input, base_img, val_range):
    conv_result = net.forward_generate(input)
    channel_loss = torch.sum(torch.square(conv_result[:, faulted_channel] - empty_channel))
    ssim_loss = 1 - pytorch_msssim.ssim(input, base_img, val_range=val_range)
    return ssim_loss + channel_loss, channel_loss


def get_confidence(output):
    # Apply the softmax function to obtain the probability distribution over the classes
    softmax_output = torch.nn.functional.softmax(output, dim=1)

    # Get the index of the predicted class
    predicted_class_index = torch.argmax(output)

    return predicted_class_index, softmax_output[0][predicted_class_index]


def save_image(input, dir, name):
    # save the generated image
    generated_img = inverse_normalize(input[0])
    # clamp into 0-1 range
    generated_img = torch.clamp(generated_img, 0, 1)
    # convert to numpy array
    img_to_save = generated_img.detach().cpu().numpy().transpose(1, 2, 0)
    plt.imsave(f"{dir}/{name}.png", img_to_save)


def validate_exploitability(input, base_img, i, target_class):
    '''
    Checks whether the generated image can exploit the network. It accounts for loss of bit
    precision that occurs during inverse normalization and saving the image to png.
    '''
    # save the generated image
    save_image(input, "fooling_images", "test")

    # read the image from png to pytorch
    generated_img = torchvision.io.read_image(f"fooling_images/test.png")[:3, :, :] / 255.0

    # normalize the image again
    generated_img = normalize(generated_img).reshape(1, 3, 32, 32)

    # forward pass
    with torch.no_grad():
        output = net(generated_img)

    pred, confidence = get_confidence(output)

    return pred.item() == target_class, confidence.item()


def validate_stealthiness(input, original_class):
    '''
    Checks whether the generated image can be correctly classified by the validation model.
    '''
    # forward pass
    with torch.no_grad():
        output = net_valid(input)

    # get the index of the max log-probability
    pred, confidence = get_confidence(output)

    return pred.item() == original_class, confidence.item()


def test_fault_simulation(x, y, attack_config):
    '''
    Tests whether the fault simulation is successful.
    '''
    # forward pass
    with torch.no_grad():
        output = net.forward(x, y, attack_config, True)

    # get the index of the max log-probability
    pred, confidence = get_confidence(output)


def update_metrics(metrics, exploit_succesful, validation_successful, below_threshold, original_class, target_class):
    if original_class == target_class:
        metrics["target_class_members"] += 1
        return metrics

    if exploit_succesful and validation_successful:
        metrics["fooling_and_validation_successful"] += 1
    elif exploit_succesful and below_threshold:
        metrics["fooling_successful_below_thresh"] += 1
    elif exploit_succesful:
        metrics["fooling_successful"] += 1
    else:
        metrics["fooling_unsuccessful"] += 1

    return metrics


def print_final_metrics(metrics):
    global SAMPLE_SIZE
    fooling_images = SAMPLE_SIZE - metrics["target_class_members"]
    print(f"Fooling successful: {(fooling_images - metrics['fooling_unsuccessful']) / fooling_images * 100:.2f}%")
    print(
        f"Fooling successful and validation successful: {metrics['fooling_and_validation_successful'] / fooling_images * 100:.2f}%")
    print(
        f"Fooling successful but below confidence threshold: {metrics['fooling_successful_below_thresh'] / fooling_images * 100:.2f}%")
    print(f"Fooling unsuccessful: {metrics['fooling_unsuccessful'] / fooling_images * 100:.2f}%")


for i in range(SAMPLE_SIZE):
    base_img = base_imgs[i].reshape(1, 3, 32, 32)
    input = base_img.clone()

    input.requires_grad = True
    base_img.requires_grad = False

    val_range = float(base_img.max() - base_img.min())

    # define optimizer
    optimizer = torch.optim.Adam([input], lr=0.01)

    # run optimization
    for j in range(1000):
        optimizer.zero_grad()
        total_loss, channel_loss = loss(input, base_img, val_range)
        total_loss.backward()
        optimizer.step()
        if j % 10 == 0:
            # test_fault_simulation(input, labels[i], fault_config)
            exploit_successful, confidence = validate_exploitability(input, base_img, i, target_class)
            if exploit_successful and confidence > CONFIDENCE_THRESH:
                break

    below_thresh = False
    exploit_successful, confidence = validate_exploitability(input, base_img, i, target_class)
    if exploit_successful and confidence > CONFIDENCE_THRESH:
        print(
            f"Image {i} is exploiting the fault successfully with >{CONFIDENCE_THRESH * 100}% confidence. "
            f"Original class: {labels[i]}, predicted class: {target_class}. Confidence: {confidence}\n")
    elif exploit_successful:
        print(
            f"Image {i} is exploiting the fault successfully, but does not meet the {CONFIDENCE_THRESH * 100}% confidence threshold. \n"
            f"Fooling confidence: {confidence * 100}%. Original class: {labels[i]} \n ")
        below_thresh = True
    else:
        print(f"Image {i} was unable to exploit the network. Original class: {labels[i]}\n")

    # check whether the generated image can be correctly classified by the validation model
    validation_successful, confidence = validate_stealthiness(input, labels[i])
    if validation_successful:
        print(f"Image {i} is stealthy. Original class: {labels[i]}, confidence: {confidence} \n")
    else:
        print(f"Image {i} is not stealthy. Original class: {labels[i]}\n")

    if exploit_successful and validation_successful:
        save_image(input, "fooling_images/fooling_and_validation_successful", f"fool_{i}_class_{target_class}")
    elif exploit_successful:
        save_image(input, "fooling_images/fooling_successful", f"fool_{i}_class_{target_class}")
    else:
        save_image(input, "fooling_images/fooling_unsuccessful", f"fool_{i}_class_{target_class}")

    update_metrics(metrics, exploit_successful, validation_successful, below_thresh, labels[i], target_class)

print_final_metrics(metrics)