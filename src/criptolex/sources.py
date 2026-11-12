"""Registry of the official sources used by the assistant.

Every source must be public, official and citable. Third-party blogs are
deliberately excluded: the value of this project is citation accuracy.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    key: str
    title: str
    issuer: str
    url: str
    kind: str  # "html" | "pdf" | "bcn_json"
    year: int
    legal: bool  # True when the text is structured as articles
    api_url: str | None = None


SOURCES: tuple[Source, ...] = (
    Source(
        key="ley_21521",
        title="Ley 21.521 — Ley Fintech",
        issuer="Biblioteca del Congreso Nacional (BCN)",
        url="https://www.bcn.cl/leychile/navegar?idNorma=1187323",
        kind="bcn_txt",
        year=2023,
        legal=True,
        api_url=(
            "https://nuevo.leychile.cl/servicios/Consulta/Exportar"
            "?radioExportar=Normas&exportar_formato=txt"
            "&nombrearchivo=Ley-21521_2023-02-03"
            "&exportar_con_notas_bcn=False&exportar_con_notas_originales=False"
            "&exportar_con_notas_al_pie=False"
            "&hddResultadoExportar=1187323.2023-02-03.0.0%23"
        ),
    ),
    Source(
        key="ley_19913",
        title="Ley 19.913 — Lavado de activos y UAF",
        issuer="Biblioteca del Congreso Nacional (BCN)",
        url="https://www.bcn.cl/leychile/navegar?idNorma=219119",
        kind="bcn_txt",
        year=2003,
        legal=True,
        api_url=(
            "https://nuevo.leychile.cl/servicios/Consulta/Exportar"
            "?radioExportar=Normas&exportar_formato=txt"
            "&nombrearchivo=LEY-19913_2026-05-30"
            "&exportar_con_notas_bcn=False&exportar_con_notas_originales=False"
            "&exportar_con_notas_al_pie=False"
            "&hddResultadoExportar=219119.2026-05-30.0.0%23"
        ),
    ),
    Source(
        key="ncg_502_cmf",
        title="NCG 502 — Registro y obligaciones de prestadores Fintech",
        issuer="Comisión para el Mercado Financiero (CMF)",
        url="https://www.cmfchile.cl/normativa/ncg_502_2024.pdf",
        kind="pdf",
        year=2024,
        legal=False,
    ),
    Source(
        key="circular_62_uaf",
        title="Circular 62 — Instrucciones generales UAF",
        issuer="Unidad de Análisis Financiero (UAF)",
        url="https://www.uaf.cl/media/documentos/Circular_N62.pdf",
        kind="pdf",
        year=2025,
        legal=False,
    ),
    Source(
        key="sii_circular_43_2021",
        title="Circular 43/2021 — Tratamiento tributario de criptomonedas",
        issuer="Servicio de Impuestos Internos (SII)",
        url="https://www.sii.cl/normativa_legislacion/circulares/2021/circu43.pdf",
        kind="pdf",
        year=2021,
        legal=False,
    ),
    Source(
        key="sii_faq_cripto",
        title="Preguntas frecuentes — Activos digitales y criptomonedas",
        issuer="Servicio de Impuestos Internos (SII)",
        url="https://www.sii.cl/preguntas_frecuentes/criptomonedas/arbol_faqs_criptomonedas_1653.htm",
        kind="sii_faq",
        year=2026,
        legal=False,
    ),
    Source(
        key="sii_dj1964",
        title="DJ 1964 — Declaración jurada de activos digitales",
        issuer="Servicio de Impuestos Internos (SII)",
        url="https://www.sii.cl/ayudas/ayudas_por_servicios/renta/2026/instrucciones_dj1964.pdf",
        kind="pdf",
        year=2026,
        legal=False,
    ),
)

BY_KEY = {source.key: source for source in SOURCES}
