from models.fl_detection_model import LandmarkDetectionModel
from models.multilabel_classification_model import MultiLabelClassificationModel
from models.single_label_classification_model import SingleLabelClassificationModel
from client_handling.client import Client
import torchvision
from client_handling.seed import set_seed

"""
Helper script to streamline the retrieval of clients.
"""


def _create_cifar_clients(
    title, t_an, t_ob, dm_cifar, an_client_ids, ob_client_ids, device, backbone_layers,
):
    """Creates clients for the cifar dataset.

    Args:
        title (str): Title of the experiment
        t_an (str): Name of the tasktype animals
        t_ob (str): Name of the tasktype objects
        dm_cifar (str): The datamanager
        an_client_ids ([str]): The animal client ids.
        ob_client_ids ([srt]): The object client ids.
        device (torch.device): The device, gpu or cpu.
        backbone_layers (str): Size of backbone.

    Returns:
        [an_cifar_clients, ob_cifar_clients]: The generated clients.
        Some focusing on the animal classification task and some which focus on the object classification task.
    """
    an_cifar_clients = []
    ob_cifar_clients = []

    # Create animal clients
    for idx, id in enumerate(an_client_ids):
        seed = idx + 300
        set_seed(seed)
        an_cifar_clients.append(
            Client(
                c_id=id,
                title=title,
                tasktype=t_an,
                seed=seed,
                model=SingleLabelClassificationModel(
                    num_classes=6,
                    model=torchvision.models.resnet18(pretrained=False),
                    train_loader=dm_cifar.train_animals_loaders[idx],
                    val_loader=dm_cifar.val_animals_loaders[idx],
                    test_loader=dm_cifar.test_animals_loader,
                    backbone_layers=backbone_layers,
                ).to(device),
            )
        )

    # Create object clients.
    for idx, id in enumerate(ob_client_ids):
        seed = idx + 400
        set_seed(seed)
        ob_cifar_clients.append(
            Client(
                c_id=id,
                title=title,
                tasktype=t_ob,
                seed=seed,
                model=SingleLabelClassificationModel(
                    num_classes=4,
                    model=torchvision.models.resnet18(pretrained=False),
                    train_loader=dm_cifar.train_objects_loaders[idx],
                    val_loader=dm_cifar.val_objects_loaders[idx],
                    test_loader=dm_cifar.test_objects_loader,
                    backbone_layers=backbone_layers,
                ).to(device),
            )
        )

    return an_cifar_clients, ob_cifar_clients


def _create_celeba_clients(
    title,
    t_multilabel,
    t_facial_landmarks,
    dm_celeba,
    ml_client_ids,
    fl_client_ids,
    device,
    backbone_layers,
):
    """Creates clients for the CelebA dataset.

    Args:
        title (str): Title of the experiment
        t_multilabel (str): Name of the multilabel task
        t_facial_landmarks (str): Name of facial landmark detection task
        dm_celeba (str): The datamanager
        ml_client_ids ([str]): the multi label client ids
        fl_client_ids ([str]): the facial landmark detection client ids
        device (torch.device): The device, gpu or cpu.
        backbone_layers (str): Size of backbone.

    Returns:
        [ml_celeba_clients, fl_celeba_clients]: The generated clients.
        Some focusing on the multi label classification task and some which focus on the facial landmark detection task.
    """
    ml_celeba_clients = []
    fl_celeba_clients = []
    for idx, id in enumerate(ml_client_ids):
        seed = idx + 100
        set_seed(seed)
        ml_celeba_clients.append(
            Client(
                c_id=id,
                title=title,
                tasktype=t_multilabel,
                seed=seed,
                model=MultiLabelClassificationModel(
                    num_classes=40,
                    model=torchvision.models.resnet18(pretrained=False),
                    train_loader=dm_celeba.train_multilabel_loaders[idx],
                    val_loader=dm_celeba.val_multilabel_loaders[idx],
                    test_loader=dm_celeba.test_multilabel_loader,
                    backbone_layers=backbone_layers,
                ).to(device),
            )
        )

    for idx, id in enumerate(fl_client_ids):
        seed = idx + 200
        set_seed(seed)
        fl_celeba_clients.append(
            Client(
                c_id=id,
                title=title,
                tasktype=t_facial_landmarks,
                seed=seed,
                model=LandmarkDetectionModel(
                    num_landmarks=5,
                    model=torchvision.models.resnet18(pretrained=False),
                    train_loader=dm_celeba.train_landmarks_loaders[idx],
                    val_loader=dm_celeba.val_landmarks_loaders[idx],
                    test_loader=dm_celeba.test_landmarks_loader,
                    backbone_layers=backbone_layers,
                ).to(device),
            )
        )

    return ml_celeba_clients, fl_celeba_clients


def get_clients(
    dataset,
    exp_title,
    task_1,
    task_2,
    data_manager,
    task_1_clients_ids,
    task_2_clients_ids,
    device,
    backbone_layers,
):
    """Decides which clients to create based on the provided dataset. It then calls one of the above two defined functions.

    Args:
        dataset (str): The name of the dataset
        exp_title (str): The title of the experiment
        task_1 (str): The name of the first task group
        task_2 (srt): The name of the second task group
        data_manager (Datamanager): The datamanager
        task_1_clients_ids ([str]): The client ids belonging to the first task group
        task_2_clients_ids ([srt]): The client ids belonging to the second task group
        device (torch.device): The device, gpu or cpu.
        backbone_layers (str): Size of backbone.

    Returns:
        [clients_t1, clients_t2]: The generated clients for each task group.
    """
    print("Getting Clients")
    if dataset == "cifar10":
        return _create_cifar_clients(
            exp_title,
            task_1,
            task_2,
            data_manager,
            task_1_clients_ids,
            task_2_clients_ids,
            device,
            backbone_layers,
        )
    elif dataset == "celeba":
        return _create_celeba_clients(
            exp_title,
            task_1,
            task_2,
            data_manager,
            task_1_clients_ids,
            task_2_clients_ids,
            device,
            backbone_layers,
        )
    else:
        raise "Dataset not properly defined!"
