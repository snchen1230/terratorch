from torchgeo.datasets import VHR10
from geobench_v2.datasets import GeoBenchWindTurbine
import pandas as pd
import numpy as np

from torchgeo.datasets.utils import (
    Path,
    check_integrity,
    download_and_extract_archive,
    download_url,
    lazy_import,
    percentile_normalization,
)

from collections.abc import Callable
from typing import Any, ClassVar

from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import patches
import pdb


class mWindTurbine(GeoBenchWindTurbine):   
    """mWindTurbine dataset.
    The WindTurbine dataset from geobench_v2.
    """
    categories = ("background", "wind_turbine", "wind_turbine2")
  
    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
        show_feats: str | None = 'boxes',
        box_alpha: float = 0.7,
        mask_alpha: float = 0.7,
        confidence_score = 0.5
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            suptitle: optional string to use as a suptitle
            show_titles: flag indicating whether to show titles above each panel
            show_feats: optional string to pick features to be shown: boxes, masks, both
            box_alpha: alpha value of box
            mask_alpha: alpha value of mask

        Returns:
            a matplotlib Figure with the rendered sample

        Raises:
            AssertionError: if ``show_feats`` argument is invalid
            DependencyNotFoundError: If plotting masks and scikit-image is not installed.

        .. versionadded:: 0.4
        """
        assert show_feats in {'boxes', 'masks', 'both'}

        image = percentile_normalization(sample['image'].permute(1, 2, 0).numpy())

        if self.split == 'negative':
            fig, axs = plt.subplots(squeeze=False)
            axs[0, 0].imshow(image)
            axs[0, 0].axis('off')

            if suptitle is not None:
                plt.suptitle(suptitle)
            return fig

        if show_feats != 'boxes':
            skimage = lazy_import('skimage')

        boxes = sample['boxes'].cpu().numpy()
        labels = sample['labels'].cpu().numpy()

        if 'masks' in sample:
            masks = [mask.squeeze().cpu().numpy() for mask in sample['masks']]

        n_gt = len(boxes)

        ncols = 1
        show_predictions = 'prediction_labels' in sample

        if show_predictions:
            show_pred_boxes = False
            show_pred_masks = False
            prediction_labels = sample['prediction_labels'].numpy()
            prediction_scores = sample['prediction_scores'].numpy()
            if 'prediction_boxes' in sample:
                prediction_boxes = sample['prediction_boxes'].numpy()
                show_pred_boxes = True
            if 'prediction_masks' in sample:
                prediction_masks = sample['prediction_masks'].numpy()
                show_pred_masks = True

            n_pred = len(prediction_labels)
            ncols += 1

        # Display image
        fig, axs = plt.subplots(ncols=ncols, squeeze=False, figsize=(ncols * 10, 13))
        axs[0, 0].imshow(image)
        axs[0, 0].axis('off')

        cm = plt.get_cmap('gist_rainbow')
        for i in range(n_gt):
            class_num = labels[i]
            color = cm(class_num / len(self.categories))

            # Add bounding boxes
            x1, y1, x2, y2 = boxes[i]
            if show_feats in {'boxes', 'both'}:
                r = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    alpha=box_alpha,
                    linestyle='dashed',
                    edgecolor=color,
                    facecolor='none',
                )
                axs[0, 0].add_patch(r)

            # Add labels
            label = self.categories[class_num]
            caption = label
            axs[0, 0].text(
                x1, y1 - 8, caption, color='white', size=11, backgroundcolor='none'
            )

            # Add masks
            if show_feats in {'masks', 'both'} and 'masks' in sample:
                mask = masks[i]
                contours = skimage.measure.find_contours(mask, 0.5)
                for verts in contours:
                    verts = np.fliplr(verts)
                    p = patches.Polygon(
                        verts, facecolor=color, alpha=mask_alpha, edgecolor='white'
                    )
                    axs[0, 0].add_patch(p)

            if show_titles:
                axs[0, 0].set_title('Ground Truth')

        if show_predictions:
            axs[0, 1].imshow(image)
            axs[0, 1].axis('off')
            for i in range(n_pred):
                score = prediction_scores[i]
                if score < confidence_score:
                    continue

                class_num = prediction_labels[i]
                color = cm(class_num / len(self.categories))

                if show_pred_boxes:
                    # Add bounding boxes
                    x1, y1, x2, y2 = prediction_boxes[i]
                    r = patches.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=2,
                        alpha=box_alpha,
                        linestyle='dashed',
                        edgecolor=color,
                        facecolor='none',
                    )
                    axs[0, 1].add_patch(r)

                    # Add labels
                    label = self.categories[class_num]
                    caption = f'{label} {score:.3f}'
                    axs[0, 1].text(
                        x1,
                        y1 - 8,
                        caption,
                        color='white',
                        size=11,
                        backgroundcolor='none',
                    )

                # Add masks
                if show_pred_masks:

                    mask = prediction_masks[i][0]
                    contours = skimage.measure.find_contours(mask, 0.5)
                    for verts in contours:
                        verts = np.fliplr(verts)
                        p = patches.Polygon(
                            verts, facecolor=color, alpha=mask_alpha, edgecolor='white'
                        )
                        axs[0, 1].add_patch(p)

            if show_titles:
                axs[0, 1].set_title('Prediction')

        if suptitle is not None:
            plt.suptitle(suptitle)

        plt.tight_layout()

        return fig
