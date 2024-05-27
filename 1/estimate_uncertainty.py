import torch


def estimate_uncertainty(model, image, n_passes = 10):
    model.eval()
    predictions_label1 = []
    predictions_label2 = []

    with torch.no_grad():
        for _ in range(n_passes):
            label1_pred, label2_pred = model(image, apply_dropout=True)
            predictions_label1.append(label1_pred)
            predictions_label2.append(label2_pred)

    uncertainty_label1 = torch.std(torch.stack(predictions_label1), dim=0)
    uncertainty_label2 = torch.std(torch.stack(predictions_label2), dim=0)

    return uncertainty_label1, uncertainty_label2