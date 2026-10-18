"""Anatomically matched MUSE MRI and AAL PET node definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegionNode:
    """Map one anatomical graph node to its MRI and PET source labels."""

    name: str
    hemisphere: str
    structure: str
    muse_region: str
    aal_region: str

    @property
    def mri_feature(self) -> str:
        """Return the extracted ICV-normalised MUSE volume column."""

        return f"mri_{self.muse_region}_icv_fraction"

    @property
    def pet_feature(self) -> str:
        """Return the extracted cerebellar-reference AAL PET column."""

        return f"pet_{self.aal_region}_cerebellar_ratio"


# These pairs have clear left/right anatomical counterparts in MUSE and AAL.
# Atlas boundaries are not identical, so the mapping is documented as
# anatomically corresponding rather than voxel-for-voxel equivalent.
_PAIRED_REGION_SPECS: tuple[tuple[str, str, str], ...] = (
    ("amygdala", "amygdala", "Amygdala"),
    ("hippocampus", "hippocampus", "Hippocampus"),
    ("angular gyrus", "ang_angular_gyrus", "Angular"),
    ("fusiform gyrus", "fug_fusiform_gyrus", "Fusiform"),
    ("inferior temporal gyrus", "itg_inferior_temporal_gyrus", "Temporal_Inf"),
    ("middle temporal gyrus", "mtg_middle_temporal_gyrus", "Temporal_Mid"),
    ("posterior cingulate gyrus", "pcgg_posterior_cingulate_gyrus", "Cingulum_Post"),
    ("precuneus", "pcu_precuneus", "Precuneus"),
    ("parahippocampal gyrus", "phg_parahippocampal_gyrus", "ParaHippocampal"),
    ("superior parietal lobule", "spl_superior_parietal_lobule", "Parietal_Sup"),
    ("caudate", "caudate", "Caudate"),
    ("putamen", "putamen", "Putamen"),
    ("pallidum", "pallidum", "Pallidum"),
    ("thalamus", "thalamus_proper", "Thalamus"),
    ("anterior cingulate gyrus", "acgg_anterior_cingulate_gyrus", "Cingulum_Ant"),
    ("calcarine cortex", "calc_calcarine_cortex", "Calcarine"),
    ("cuneus", "cun_cuneus", "Cuneus"),
    ("gyrus rectus", "gre_gyrus_rectus", "Rectus"),
    ("inferior occipital gyrus", "iog_inferior_occipital_gyrus", "Occipital_Inf"),
    ("lingual gyrus", "lig_lingual_gyrus", "Lingual"),
    ("middle cingulate gyrus", "mcgg_middle_cingulate_gyrus", "Cingulum_Mid"),
    ("middle frontal gyrus", "mfg_middle_frontal_gyrus", "Frontal_Mid"),
    ("middle occipital gyrus", "mog_middle_occipital_gyrus", "Occipital_Mid"),
    (
        "opercular inferior frontal gyrus",
        "opifg_opercular_part_of_the_inferior_frontal_gyrus",
        "Frontal_Inf_Oper",
    ),
    (
        "orbital inferior frontal gyrus",
        "orifg_orbital_part_of_the_inferior_frontal_gyrus",
        "Frontal_Inf_Orb",
    ),
    (
        "triangular inferior frontal gyrus",
        "trifg_triangular_part_of_the_inferior_frontal_gyrus",
        "Frontal_Inf_Tri",
    ),
    ("precentral gyrus", "prg_precentral_gyrus", "Precentral"),
    ("postcentral gyrus", "pog_postcentral_gyrus", "Postcentral"),
    ("supplementary motor cortex", "smc_supplementary_motor_cortex", "Supp_Motor_Area"),
    ("supramarginal gyrus", "smg_supramarginal_gyrus", "SupraMarginal"),
    ("superior frontal gyrus", "sfg_superior_frontal_gyrus", "Frontal_Sup"),
    ("superior occipital gyrus", "sog_superior_occipital_gyrus", "Occipital_Sup"),
    ("superior temporal gyrus", "stg_superior_temporal_gyrus", "Temporal_Sup"),
)


def _build_region_nodes() -> tuple[RegionNode, ...]:
    """Create left/right nodes in a stable, documented order."""

    nodes: list[RegionNode] = []
    for structure, muse_stem, aal_stem in _PAIRED_REGION_SPECS:
        for hemisphere, muse_prefix, aal_suffix in (
            ("left", "left", "L"),
            ("right", "right", "R"),
        ):
            nodes.append(
                RegionNode(
                    name=f"{hemisphere.title()} {structure}",
                    hemisphere=hemisphere,
                    structure=structure,
                    muse_region=f"{muse_prefix}_{muse_stem}",
                    aal_region=f"{aal_stem}_{aal_suffix}",
                )
            )
    return tuple(nodes)


REGION_NODES: tuple[RegionNode, ...] = _build_region_nodes()


def feature_columns(modality: str | None = None) -> list[str]:
    """Return ordered feature columns for MRI, PET, or both modalities."""

    if modality == "MRI":
        return [region.mri_feature for region in REGION_NODES]
    if modality == "PET":
        return [region.pet_feature for region in REGION_NODES]
    if modality is None:
        return feature_columns("MRI") + feature_columns("PET")
    raise ValueError("modality must be 'MRI', 'PET', or None")


def region_feature_map() -> dict[str, tuple[str, str]]:
    """Return region name to ordered (MRI, PET) feature mapping."""

    return {region.name: (region.mri_feature, region.pet_feature) for region in REGION_NODES}
