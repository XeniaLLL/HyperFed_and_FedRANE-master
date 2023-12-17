import torch
import torch.nn.functional

def pairwise_distances(
        a: torch.Tensor,
        b: torch.Tensor,
        p: float = 2.0,
        eps: float = 0.0
    ) -> torch.Tensor:
    """
    :param a: shape(m, d)
    :param b: shape(n, d)
    :return: shape(m, n)
    """
    a = a.unsqueeze(dim=1)  # shape(m, 1, d)
    b = b.unsqueeze(dim=0)  # shape(1, n, d)
    result = torch.abs(a - b + eps).pow(p)  # shape(m, n, d)
    result = result.sum(dim=2)  # shape(m, n)
    return result.pow(1 / p)


def students_t_kernel(dist: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    :param d: distances between samples and centroids, shape(N, cluster_num)
    :return: label distribution q, shape(N, cluster_num)
    """
    numerator = (1 + dist / alpha).pow(- (alpha + 1) / 2)  # shape(N, cluster_num)
    # normalize each row into a probability distribution
    return torch.nn.functional.normalize(numerator, p=1, dim=1)


def target_distribution(q: torch.Tensor) -> torch.Tensor:
    """
    :param q: label distribution, shape(N, cluster_num)
    :return: target distribution p, shape(N, cluster_num)
    """
    f = q.sum(dim=0)  # shape(cluster_num,)
    numerator = q ** 2 / f  # shape(N, cluster_num)
    return torch.nn.functional.normalize(numerator, p=1, dim=1)
