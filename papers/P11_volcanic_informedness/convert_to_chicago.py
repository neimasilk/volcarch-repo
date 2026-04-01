"""Convert P11 draft from natbib author-date to Chicago 17th notes-bibliography.

Creates draft_v0.3_chicago.tex with \footnote{} replacing \citep{}/\citet{}.
Then run: pandoc draft_v0.3_chicago.tex -o draft_v0.3_cornell_chicago.docx
"""
import re

# Chicago notes-bibliography: first full citation in footnote
# Format: First Last, *Title* (Place: Publisher, Year), pages.
FOOTNOTES = {
    'gvp2023': 'Global Volcanism Program, \\emph{Volcanoes of the World}, v. 5.1.1 (Smithsonian Institution, 2023), \\url{https://volcano.si.edu/}.',
    'mohr1938': 'E. C. J. Mohr, "The Relation between Soil and Population Density in the Netherlands East Indies," in \\emph{Comptes Rendus du Congr\\`{e}s International de G\\\'eographie} (Amsterdam, 1938).',
    'whitten1996': 'Tony Whitten, Roehayat Emon Soeriaatmadja, and Suraya A. Afiff, \\emph{The Ecology of Java and Bali} (Singapore: Periplus Editions, 1996).',
    'thouret1999': 'Jean-Claude Thouret, "Volcanic Geomorphology---an Overview," \\emph{Earth-Science Reviews} 47, no. 1--2 (1999): 95--131.',
    'dumarcay1993': 'Jacques Dumar\\c{c}ay, \\emph{The Temples of Java} (Oxford: Oxford University Press, 1993).',
    'soekmono1995': 'R. Soekmono, \\emph{The Javanese Candi: Function and Meaning} (Leiden: Brill, 1995).',
    'dharma2024': 'DHARMA project, \\emph{The Domestication of "Hindu" Asceticism and the Religious Making of South and Southeast Asia: Epigraphic Database} (CNRS/ERC, 2024), \\url{https://dharma.hypotheses.org/}.',
    'barnes2003': 'Gina L. Barnes, "Origins of the Japanese Islands: The New `Big Picture,\'" \\emph{Japan Review} 15 (2003): 3--50.',
    'takata2022': 'Hiroki Takata and Takahiro Yanase, "Production, Preservation and Dissemination of Archaeological Data in Japan," \\emph{Internet Archaeology} 58 (2022).',
    'shimoyama2002': 'Satoru Shimoyama, "Volcanic Disasters and Archaeological Sites in Southern Kyushu, Japan," in \\emph{Natural Disasters and Cultural Change}, ed. Robin Torrence and John Grattan (London: Routledge, 2002), 326--341.',
    'lavigne2003': 'Franck Lavigne and Jean-Claude Thouret, "Sediment Transportation and Deposition by Rain-Triggered Lahars at Merapi Volcano, Central Java, Indonesia," \\emph{Geomorphology} 49, no. 1--2 (2003): 45--69.',
    'schiffer1987': 'Michael B. Schiffer, \\emph{Formation Processes of the Archaeological Record} (Albuquerque: University of New Mexico Press, 1987).',
    'sheets2002': 'Payson D. Sheets, ed., \\emph{Before the Volcano Erupted: The Ancient Cer\\\'en Village in Central America} (Austin: University of Texas Press, 2002).',
    'abbas2016': 'Novida Abbas, ed., \\emph{Liangan: Mozaik Peradaban Mataram Kuno di Lereng Sindoro} (Yogyakarta: Kepel Press / Balai Arkeologi Yogyakarta, 2016).',
}

# Chicago bibliography format (end of paper)
BIBLIOGRAPHY = r"""
\section*{Bibliography}

\noindent Abbas, Novida, ed. \emph{Liangan: Mozaik Peradaban Mataram Kuno di Lereng Sindoro}. Yogyakarta: Kepel Press / Balai Arkeologi Yogyakarta, 2016.

\vspace{0.5em}
\noindent Barnes, Gina L. ``Origins of the Japanese Islands: The New `Big Picture.'\,'' \emph{Japan Review} 15 (2003): 3--50.

\vspace{0.5em}
\noindent DHARMA project. \emph{The Domestication of ``Hindu'' Asceticism and the Religious Making of South and Southeast Asia: Epigraphic Database}. CNRS/ERC, 2024. \url{https://dharma.hypotheses.org/}.

\vspace{0.5em}
\noindent Dumar\c{c}ay, Jacques. \emph{The Temples of Java}. Oxford: Oxford University Press, 1993.

\vspace{0.5em}
\noindent Global Volcanism Program. \emph{Volcanoes of the World}. V. 5.1.1. Smithsonian Institution, 2023. \url{https://volcano.si.edu/}.

\vspace{0.5em}
\noindent Lavigne, Franck, and Jean-Claude Thouret. ``Sediment Transportation and Deposition by Rain-Triggered Lahars at Merapi Volcano, Central Java, Indonesia.'' \emph{Geomorphology} 49, no. 1--2 (2003): 45--69.

\vspace{0.5em}
\noindent Mohr, E. C. J. ``The Relation between Soil and Population Density in the Netherlands East Indies.'' In \emph{Comptes Rendus du Congr\`{e}s International de G\'{e}ographie}. Amsterdam, 1938.

\vspace{0.5em}
\noindent Schiffer, Michael B. \emph{Formation Processes of the Archaeological Record}. Albuquerque: University of New Mexico Press, 1987.

\vspace{0.5em}
\noindent Sheets, Payson D., ed. \emph{Before the Volcano Erupted: The Ancient Cer\'{e}n Village in Central America}. Austin: University of Texas Press, 2002.

\vspace{0.5em}
\noindent Shimoyama, Satoru. ``Volcanic Disasters and Archaeological Sites in Southern Kyushu, Japan.'' In \emph{Natural Disasters and Cultural Change}, edited by Robin Torrence and John Grattan, 326--341. London: Routledge, 2002.

\vspace{0.5em}
\noindent Soekmono, R. \emph{The Javanese Candi: Function and Meaning}. Leiden: Brill, 1995.

\vspace{0.5em}
\noindent Takata, Hiroki, and Takahiro Yanase. ``Production, Preservation and Dissemination of Archaeological Data in Japan.'' \emph{Internet Archaeology} 58 (2022).

\vspace{0.5em}
\noindent Thouret, Jean-Claude. ``Volcanic Geomorphology---an Overview.'' \emph{Earth-Science Reviews} 47, no. 1--2 (1999): 95--131.

\vspace{0.5em}
\noindent Whitten, Tony, Roehayat Emon Soeriaatmadja, and Suraya A. Afiff. \emph{The Ecology of Java and Bali}. Singapore: Periplus Editions, 1996.
"""

def convert():
    with open('draft_v0.3.tex', 'r', encoding='utf-8') as f:
        tex = f.read()

    # Track which keys have been cited (for short-form subsequent citations)
    cited = set()

    def replace_citep(match):
        """Replace \citep{key} or \citep{key1, key2} with footnote(s)."""
        keys_str = match.group(1)
        keys = [k.strip() for k in keys_str.split(',')]
        parts = []
        for key in keys:
            if key in FOOTNOTES:
                parts.append(FOOTNOTES[key])
                cited.add(key)
            else:
                parts.append(f'[MISSING: {key}]')
        return '\\footnote{' + ' '.join(parts) + '}'

    def replace_citet(match):
        """Replace \citet{key} with Author Name + footnote."""
        key = match.group(1).strip()
        # Map keys to display names for \citet
        author_names = {
            'dumarcay1993': 'Dumar\\c{c}ay',
            'soekmono1995': 'Soekmono',
            'barnes2003': 'Barnes',
            'takata2022': 'Takata and Yanase',
            'shimoyama2002': 'Shimoyama',
            'abbas2016': 'Abbas',
            'sheets2002': 'Sheets',
        }
        name = author_names.get(key, key)
        fn = FOOTNOTES.get(key, f'[MISSING: {key}]')
        cited.add(key)
        return f'{name}\\footnote{{{fn}}}'

    # Replace \citep{...} (parenthetical)
    tex = re.sub(r'\\citep\{([^}]+)\}', replace_citep, tex)
    # Replace \citet{...} (textual)
    tex = re.sub(r'\\citet\{([^}]+)\}', replace_citet, tex)

    # Remove natbib package (not needed anymore)
    tex = tex.replace(r'\usepackage{natbib}', '% natbib removed for Chicago notes-bibliography')

    # Remove \bibliographystyle
    tex = tex.replace(r'\bibliographystyle{plainnat}', '')

    # Replace the entire \begin{thebibliography}...\end{thebibliography} with Chicago bibliography
    bib_replacement = BIBLIOGRAPHY.strip()
    tex = re.sub(
        r'\\begin\{thebibliography\}\{99\}.*?\\end\{thebibliography\}',
        lambda m: bib_replacement,
        tex,
        flags=re.DOTALL
    )

    with open('draft_v0.3_chicago.tex', 'w', encoding='utf-8') as f:
        f.write(tex)

    print(f"Created draft_v0.3_chicago.tex")
    print(f"Citations converted: {len(cited)}")
    print(f"Keys cited: {sorted(cited)}")

if __name__ == '__main__':
    convert()
