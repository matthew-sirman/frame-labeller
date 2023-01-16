import os
import torch
import torch.nn as nn

import config as cfg
from model import FrameLabeller
import loader


CUDA = torch.cuda.is_available() and cfg.USE_CUDA_IF_AVAILABLE
DEVICE = "cuda" if CUDA else "cpu"


def load_model_from_cfg(semlinks):
    """
    Load a FrameLabeller instance using the config model parameters.

    Parameters:
    semlinks (SemLinkDataset): Pre-loaded dataset of semlinks

    Returns:
    FrameLabeller: Frame labeller GNN model
    """
    model = FrameLabeller(semlinks,
                          cfg.MODEL_EMBEDDING_SIZE,
                          cfg.MODEL_ATTENTION_HEADS,
                          cfg.MODEL_HIDDEN_NODE_FEATURE_SIZE)

    model.to(DEVICE)

    return model


def load_model_params(model):
    """
    Load model data from path in config file.

    Parameters:
    model (FrameLabeller): Model to load parameters into
    """
    model_path = cfg.LOAD_MODEL

    if model_path is None:
        print("WARNING: Model not loaded! No path given")

    model.load_state_dict(torch.load(os.path.join(model_path, cfg.MODEL_NAME)))


def save_model_params(model):
    """
    Save model data to path in config file.

    Parameters:
    model (FrameLabeller): Model to save parameters from
    """
    model_path = cfg.SAVE_MODEL

    if model_path is None:
        print("WARNING: Model not saved! No path given")
        return

    os.makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_path, cfg.MODEL_NAME))


def evaluate(semlinks, model):
    """Evaluate model."""
    frame_scores = {frame: {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "support": 0} for frame in semlinks.ix2frame}
    role_scores = {role: {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "support": 0} for role in semlinks.ix2role}

    def update_scores(score_dict, true, pred):
        for entry in score_dict:
            if true == pred:
                if pred == entry:
                    score_dict[entry]["tp"] += 1
                else:
                    score_dict[entry]["tn"] += 1
            else:
                if true == entry:
                    score_dict[entry]["fn"] += 1
                elif pred == entry:
                    score_dict[entry]["fp"] += 1
                else:
                    score_dict[entry]["tn"] += 1

    root_accuracy = 0
    frame_accuracy = 0
    role_accuracy = 0
    role_total = 0

    for i in range(len(semlinks)):
        feature, true_root, true_frame, true_roles = semlinks[i]
        (pred_root, pred_frame), pred_roles = model(feature)

        pred_root = torch.argmax(pred_root)
        pred_frame = torch.argmax(pred_frame)
        pred_roles = torch.argmax(pred_roles, dim=1)

        frame_scores[true_frame.item()]["support"] += 1

        if pred_root == true_root:
            root_accuracy += 1

        update_scores(frame_scores, true_frame, pred_frame)
        if pred_frame == true_frame:
            frame_accuracy += 1

        adjacent_edges = torch.argwhere(feature["edge-in"].edge_index[0] == pred_root)
        for edge in adjacent_edges:
            update_scores(role_scores, true_roles[edge], pred_roles[edge])
            if true_roles[edge] == pred_roles[edge]:
                role_accuracy += 1
            role_total += 1
            role_scores[true_roles[edge].item()]["support"] += 1

    root_accuracy /= len(semlinks)
    frame_accuracy /= len(semlinks)
    role_accuracy /= role_total

    def f_score(beta, scores, size):
        avg_f = 0
        for entry in scores:
            tp = scores[entry]["tp"]
            fp = scores[entry]["fp"]
            fn = scores[entry]["fn"]
            p = tp / (tp + fp) if tp + fp != 0 else 0
            r = tp / (tp + fn) if tp + fn != 0 else 0

            d = beta * beta * p + r
            f = (1 + beta * beta) * (p * r) / d if d != 0 else 0
            weight = scores[entry]["support"] / size
            avg_f += f * weight

        return avg_f

    def frame_f_score(beta):
        return f_score(beta, frame_scores, len(semlinks))

    def role_f_score(beta):
        return f_score(beta, role_scores, role_total)

    return root_accuracy, frame_accuracy, role_accuracy, frame_f_score, role_f_score


def train_model():
    """Train Frame labeller model using config parameters."""
    print("Loading semlinks...")
    semlinks = loader.construct_semlink_dataset_from_config()
    print("Loaded")
    model = load_model_from_cfg(semlinks)

    if cfg.LOAD_MODEL is not None:
        load_model_params(model)

    optim = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    loss_fn = nn.NLLLoss()
    role_loss_fn = nn.NLLLoss(semlinks.compute_role_loss_weights())

    semlinks.view_data_indexed()
    semlinks.activate_train_set()

    steps = 0
    steps_loss = 0
    steps_role_loss = 0

    for epoch in range(cfg.TRAIN_EPOCHS):
        epoch_loss = 0
        epoch_role_loss = 0

        for i in range(len(semlinks)):
            feature, root_node, target_frame, target_roles = semlinks[i]
            steps += 1

            (pred_root, pred_frame), pred_roles = model(feature, root_node, teacher_force_ratio=0.5)

            frame_loss = loss_fn(pred_root, root_node) + loss_fn(pred_frame, target_frame)
            role_loss = role_loss_fn(pred_roles, target_roles)
            loss = frame_loss + role_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item()
            epoch_role_loss += role_loss.item()
            steps_loss += loss.item()
            steps_role_loss += role_loss.item()

            log_round_step = steps % cfg.TRAIN_LOG_PER_N_STEPS
            if cfg.TRAIN_LOG_PER_N_STEPS > 0 and log_round_step == 0:
                avg_loss = steps_loss / cfg.TRAIN_LOG_PER_N_STEPS
                avg_role_loss = steps_role_loss / cfg.TRAIN_LOG_PER_N_STEPS

                print(f"Step {steps}/{len(semlinks) * cfg.TRAIN_EPOCHS}. "
                      f"Average loss: {avg_loss} "
                      f"Average role loss: {avg_role_loss}")
                steps_loss = 0
                steps_role_loss = 0

            save_round_step = steps % cfg.SAVE_AFTER_STEPS
            if cfg.SAVE_AFTER_STEPS > 0 and save_round_step == 0:
                print("Saving model")
                save_model_params(model)

        print(f"Epoch {epoch + 1}/{cfg.TRAIN_EPOCHS} complete. "
              f"Average epoch loss: {epoch_loss / len(semlinks)} "
              f"Average epoch role loss: {epoch_role_loss / len(semlinks)}")

        if (epoch + 1) % cfg.VALIDATE_AFTER_EPOCHS == 0:
            print("Evaluating model on development set")

            semlinks.activate_dev_set()

            root_acc, frame_acc, role_acc, frame_f, role_f = evaluate(semlinks, model)
            print(f"Root accuracy: {root_acc}")
            print(f"Frame accuracy: {frame_acc}")
            print(f"Role accuracy: {role_acc}")
            print(f"Frame F_1 score: {frame_f(1)}, F_0.5 score: {frame_f(0.5)}")
            print(f"Role F_1 score: {role_f(1)}, F_0.5 score: {role_f(0.5)}")

            semlinks.activate_train_set()

    save_model_params(model)

    print("Evaluating model on development set")

    semlinks.activate_dev_set()

    root_acc, frame_acc, role_acc, frame_f, role_f = evaluate(semlinks, model)
    print(f"Root accuracy: {root_acc}")
    print(f"Frame accuracy: {frame_acc}")
    print(f"Role accuracy: {role_acc}")
    print(f"Frame F_1 score: {frame_f(1)}, F_0.5 score: {frame_f(0.5)}")
    print(f"Role F_1 score: {role_f(1)}, F_0.5 score: {role_f(0.5)}")
    print()

    print("Evaluating model on test set")

    semlinks.activate_test_set()

    root_acc, frame_acc, role_acc, frame_f, role_f = evaluate(semlinks, model)

    print(f"Root accuracy: {root_acc}")
    print(f"Frame accuracy: {frame_acc}")
    print(f"Role accuracy: {role_acc}")
    print(f"Frame F_1 score: {frame_f(1)}, F_0.5 score: {frame_f(0.5)}")
    print(f"Role F_1 score: {role_f(1)}, F_0.5 score: {role_f(0.5)}")
