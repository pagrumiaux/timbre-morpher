"""Audio quality metrics for evaluating morphing models."""

from __future__ import annotations

import torch
import numpy as np
import torchaudio


def si_sdr(reference: torch.Tensor, estimate: torch.Tensor, zero_mean=True) -> float:
    """Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB.

    Args:
        reference: Clean/original signal, shape (samples,) or (batch, samples).
        estimate:  Reconstructed signal, same shape.

    Returns:
        SI-SDR in dB (float). Average over batch if batched.
    """
    eps = torch.finfo(reference.dtype).eps

    # If silent, return 0
    if torch.sum(reference**2) == 0:
        return 0

    # Reshape as (batch, channels, samples)
    if len(reference.shape) == 1:
        reference = reference.unsqueeze(0)
    if len(estimate.shape) == 1:
        estimate = estimate.unsqueeze(0)
    if len(reference.shape) == 2:
        reference = reference.unsqueeze(1)
    if len(estimate.shape) == 2:
        estimate = estimate.unsqueeze(1)

    # Zero mean
    if zero_mean:
        reference = reference - torch.mean(reference, dim=-1, keepdim=True)
        estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)

    # Scaled reference
    s_target = (torch.sum(reference * estimate, dim=-1, keepdim=True)+eps) / (torch.sum(reference**2, dim=-1, keepdim=True)+eps) * reference

    # Scaled noise
    e_noise = estimate - s_target

    # SI-SDR in dB
    si_sdr = 10*torch.log10((torch.sum(s_target**2, dim=-1)+eps) / (torch.sum(e_noise**2, dim=-1)+eps))

    # Average over batch/channel
    return si_sdr.mean().item()

def multiscale_stft_loss(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    fft_sizes: list[int] = [512, 1024, 2048],
    hop_sizes: list[int] = [128, 256, 512],
    win_sizes: list[int] = [512, 1024, 2048],
) -> float:
    """Compute multi-resolution STFT loss (spectral convergence + log magnitude).

    Args:
        reference: Clean signal, shape (samples,) or (batch, 1, samples).
        estimate:  Reconstructed signal, same shape.
        fft_sizes: FFT sizes for each resolution.
        hop_sizes: Hop sizes for each resolution.
        win_sizes: Window sizes for each resolution.

    Returns:
        Averaged multi-resolution STFT loss (float). Lower is better.
    """

    # Reshape to (batch, samples)
    ref = reference.reshape(-1, reference.shape[-1]) if reference.dim() > 1 else reference
    est = estimate.reshape(-1, estimate.shape[-1]) if estimate.dim() > 1 else estimate

    n_res = len(fft_sizes)
    eps = torch.finfo(ref.dtype).eps
    loss = 0
    for fft_size, hop_size, win_size in zip(fft_sizes, hop_sizes, win_sizes):
        # STFT
        stft_ref = torch.stft(ref, n_fft=fft_size, hop_length=hop_size, win_length=win_size, window=torch.hann_window(win_size, device=ref.device), return_complex=True)
        stft_est = torch.stft(est, n_fft=fft_size, hop_length=hop_size, win_length=win_size, window=torch.hann_window(win_size, device=ref.device), return_complex=True)

        # Spectral convergence
        loss_sc = (torch.norm(torch.abs(stft_ref)-torch.abs(stft_est), p='fro')+eps) / (torch.norm(torch.abs(stft_ref), p='fro')+eps)

        # Log magnitude
        loss_mag = (1/fft_size) * torch.norm(torch.log(torch.abs(stft_ref)+eps)-torch.log(torch.abs(stft_est)+eps), p=1)

        # Total loss
        loss += loss_sc + loss_mag
    
    return loss / n_res

def frechet_audio_distance(
    real_audios: list[torch.Tensor],
    generated_audios: list[torch.Tensor],
    sample_rate: int = 44100,
    embedding_model: str = "clap",
) -> float:
    """Compute Fréchet Audio Distance between two sets of audio.

    Args:
        real_audios:      List of reference audio tensors (1D each).
        generated_audios: List of generated/reconstructed audio tensors (1D each).
        sample_rate:      Sample rate of the audio.
        embedding_model:  "clap" (LAION-CLAP, recommended) or "panns" (PANNs CNN14).

    Returns:
        FAD score (float). Lower is better.
    """
    # Extract embeddings
    extractor = _get_embedding_extractor(embedding_model)
    real_embeds = extractor(real_audios, sample_rate)
    gen_embeds = extractor(generated_audios, sample_rate)

    # Compute Fréchet distance between the two Gaussian fits
    return _frechet_distance(real_embeds, gen_embeds)


_clap_model = None
_clap_processor = None
_panns_model = None


def _get_embedding_extractor(model_name: str):
    """Return the appropriate embedding extraction function."""
    if model_name == "clap":
        return _extract_clap_embeddings
    elif model_name == "panns":
        return _extract_panns_embeddings
    else:
        raise ValueError(f"Unknown embedding model: {model_name}. Available embedding models are: 'clap', 'panns'.")


def _get_clap():
    """Load CLAP model (cached)."""
    global _clap_model, _clap_processor
    if _clap_model is None:
        from transformers import ClapModel, ClapProcessor
        _clap_model = ClapModel.from_pretrained("laion/larger_clap_music_and_speech")
        _clap_processor = ClapProcessor.from_pretrained("laion/larger_clap_music_and_speech")
        _clap_model.eval()
    return _clap_model, _clap_processor


def _extract_clap_embeddings(
    audios: list[torch.Tensor],
    sample_rate: int,
) -> np.ndarray:
    """Extract CLAP audio embeddings. Returns (N, 512) numpy array."""
    model, processor = _get_clap()

    # CLAP expects 48kHz
    resample = None
    if sample_rate != 48000:
        resample = torchaudio.transforms.Resample(sample_rate, 48000)

    embeddings = []
    with torch.no_grad():
        for audio in audios:
            # Flatten to 1D mono
            if audio.dim() > 1:
                audio = audio.reshape(-1)
            if resample is not None:
                audio = resample(audio)

            inputs = processor(
                audios=audio.cpu().numpy(),
                sampling_rate=48000,
                return_tensors="pt",
            )
            emb = model.get_audio_features(**inputs)  # (1, 512)
            embeddings.append(emb.squeeze(0).cpu().numpy())

    return np.stack(embeddings)  # (N, 512)


def _get_panns():
    """Load PANNs CNN14 model (cached)."""
    global _panns_model
    if _panns_model is None:
        from panns_inference import AudioTagging
        _panns_model = AudioTagging(checkpoint_path=None, device='cpu')
    return _panns_model


def _extract_panns_embeddings(
    audios: list[torch.Tensor],
    sample_rate: int,
) -> np.ndarray:
    """Extract PANNs CNN14 embeddings. Returns (N, 2048) numpy array."""
    model = _get_panns()

    # PANNs expects 32kHz numpy arrays
    resample = None
    if sample_rate != 32000:
        resample = torchaudio.transforms.Resample(sample_rate, 32000)

    embeddings = []
    for audio in audios:
        # Flatten to 1D mono
        if audio.dim() > 1:
            audio = audio.reshape(-1)
        if resample is not None:
            audio = resample(audio)

        audio_np = audio.cpu().numpy()[np.newaxis, :]  # (1, samples)
        _, emb = model.inference(audio_np)  # emb: (1, 2048)
        embeddings.append(emb.squeeze(0))

    return np.stack(embeddings)  # (N, 2048)


def _frechet_distance(embeds_a: np.ndarray, embeds_b: np.ndarray) -> float:
    """Compute Fréchet distance between two sets of embeddings.

    Args:
        embeds_a: (N, D) array of embeddings.
        embeds_b: (M, D) array of embeddings.

    Returns:
        Fréchet distance.
    """
    from scipy.linalg import sqrtm

    mu_a = embeds_a.mean(axis=0)
    mu_b = embeds_b.mean(axis=0)
    cov_a = np.cov(embeds_a, rowvar=False)
    cov_b = np.cov(embeds_b, rowvar=False)

    diff = mu_a - mu_b
    mean_diff_sq = np.dot(diff, diff)

    # Matrix square root — can return complex values due to numerical issues
    cov_product = cov_a @ cov_b
    sqrt_cov = sqrtm(cov_product)

    # Discard imaginary part from numerical errors
    if np.iscomplexobj(sqrt_cov):
        sqrt_cov = sqrt_cov.real

    trace = np.trace(cov_a + cov_b - 2 * sqrt_cov)

    return float(mean_diff_sq + trace)


_cdpam_model = None

def _get_cdpam():
    global _cdpam_model
    if _cdpam_model is None:
        import cdpam
        _cdpam_model = cdpam.CDPAM(dev='cpu')
    return _cdpam_model


def cdpam_distance(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    sample_rate: int = 44100,
) -> float:
    """Compute CDPAM perceptual distance between two audio signals.

    Args:
        reference: Reference audio tensor, shape (samples,).
        estimate:  Compared audio tensor, shape (samples,).
        sample_rate: Sample rate (CDPAM expects 22050Hz — resample if needed).

    Returns:
        CDPAM distance (float). Lower means more perceptually similar.
    """
    # Load CDPAM
    loss_fn = _get_cdpam()

    # Flatten to mono 1D
    if reference.dim() == 3:
        reference = reference.mean(dim=1).squeeze(0)
    elif reference.dim() == 2:
        reference = reference.squeeze(0)
    if estimate.dim() == 3:
        estimate = estimate.mean(dim=1).squeeze(0)
    elif estimate.dim() == 2:
        estimate = estimate.squeeze(0)

    # Resample if needed, CDPAM expects 22050 Hz
    if sample_rate != 22050:
        reference = torchaudio.transforms.Resample(sample_rate, 22050)(reference)
        estimate = torchaudio.transforms.Resample(sample_rate, 22050)(estimate)

    # Reshape to CDPAM expected input: (1, 1, samples)
    reference = reference.unsqueeze(0).unsqueeze(0)
    estimate = estimate.unsqueeze(0).unsqueeze(0)

    # Compute loss
    loss = loss_fn(reference, estimate).item()

    return loss
