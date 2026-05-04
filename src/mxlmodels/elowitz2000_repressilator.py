"""Repressilator (Elowitz 2000): synthetic genetic oscillator with three cyclically repressing genes.

Reference: Elowitz, M. B. and Leibler, S.
"A synthetic oscillatory network of transcriptional regulators."
Nature 403 (2000): 335-338.
DOI: 10.1038/35002125
biomodels: https://www.biomodels.org/BIOMD0000000012
"""

from mxlpy import Model


def _transcription_laci(
    alpha: float,
    alpha0: float,
    p_ci: float,
    n: float,
) -> float:
    """Transcription of lacI repressed by CI protein via Hill kinetics."""
    return alpha / (1.0 + p_ci**n) + alpha0


def _transcription_tetr(
    alpha: float,
    alpha0: float,
    p_laci: float,
    n: float,
) -> float:
    """Transcription of tetR repressed by LacI protein via Hill kinetics."""
    return alpha / (1.0 + p_laci**n) + alpha0


def _transcription_ci(
    alpha: float,
    alpha0: float,
    p_tetr: float,
    n: float,
) -> float:
    """Transcription of cI repressed by TetR protein via Hill kinetics."""
    return alpha / (1.0 + p_tetr**n) + alpha0


def _mrna_degradation(
    m: float,
) -> float:
    """First-order mRNA degradation."""
    return m


def _translation(
    beta: float,
    m: float,
) -> float:
    """Protein synthesis proportional to mRNA level."""
    return beta * m


def _protein_degradation(
    beta: float,
    p: float,
) -> float:
    """First-order protein degradation scaled by beta."""
    return beta * p


def create_model() -> Model:
    """Build the Repressilator model: synthetic three-gene ring oscillator where each repressor cyclically inhibits the next."""
    return (
        Model()
        .add_variable("MlacI", initial_value=0.0)
        .add_variable("MtetR", initial_value=0.0)
        .add_variable("McI", initial_value=0.0)
        .add_variable("PlacI", initial_value=2.0)
        .add_variable("PtetR", initial_value=1.0)
        .add_variable("PcI", initial_value=3.0)
        .add_parameter("Alpha", value=216.0)
        .add_parameter("Alpha0", value=0.216)
        .add_parameter("Beta", value=5.0)
        .add_parameter("N", value=2.0)
        .add_reaction(
            "transcription_laci",
            fn=_transcription_laci,
            args=["Alpha", "Alpha0", "PcI", "N"],
            stoichiometry={"MlacI": 1.0},
        )
        .add_reaction(
            "mrna_degradation_laci",
            fn=_mrna_degradation,
            args=["MlacI"],
            stoichiometry={"MlacI": -1.0},
        )
        .add_reaction(
            "transcription_tetr",
            fn=_transcription_tetr,
            args=["Alpha", "Alpha0", "PlacI", "N"],
            stoichiometry={"MtetR": 1.0},
        )
        .add_reaction(
            "mrna_degradation_tetr",
            fn=_mrna_degradation,
            args=["MtetR"],
            stoichiometry={"MtetR": -1.0},
        )
        .add_reaction(
            "transcription_ci",
            fn=_transcription_ci,
            args=["Alpha", "Alpha0", "PtetR", "N"],
            stoichiometry={"McI": 1.0},
        )
        .add_reaction(
            "mrna_degradation_ci",
            fn=_mrna_degradation,
            args=["McI"],
            stoichiometry={"McI": -1.0},
        )
        .add_reaction(
            "translation_laci",
            fn=_translation,
            args=["Beta", "MlacI"],
            stoichiometry={"PlacI": 1.0},
        )
        .add_reaction(
            "protein_degradation_laci",
            fn=_protein_degradation,
            args=["Beta", "PlacI"],
            stoichiometry={"PlacI": -1.0},
        )
        .add_reaction(
            "translation_tetr",
            fn=_translation,
            args=["Beta", "MtetR"],
            stoichiometry={"PtetR": 1.0},
        )
        .add_reaction(
            "protein_degradation_tetr",
            fn=_protein_degradation,
            args=["Beta", "PtetR"],
            stoichiometry={"PtetR": -1.0},
        )
        .add_reaction(
            "translation_ci",
            fn=_translation,
            args=["Beta", "McI"],
            stoichiometry={"PcI": 1.0},
        )
        .add_reaction(
            "protein_degradation_ci",
            fn=_protein_degradation,
            args=["Beta", "PcI"],
            stoichiometry={"PcI": -1.0},
        )
    )
