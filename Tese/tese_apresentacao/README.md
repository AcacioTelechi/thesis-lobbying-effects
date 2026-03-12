# Apresentação da Tese (15 min)

Slides em Beamer, português (pt-BR).

## Requisitos (Beamer)

Se aparecer `File 'beamer.cls' not found`, instale o pacote Beamer:

- **Debian/Ubuntu (TeX Live):**
  ```bash
  sudo apt-get install texlive-latex-extra
  ```
- Ou apenas Beamer e temas:
  ```bash
  sudo apt-get install texlive-latex-recommended texlive-latex-extra
  ```

## Compilação

Na pasta `Tese/tese_apresentacao/` (com citações e referências):

```bash
pdflatex apresentacao.tex
bibtex apresentacao
pdflatex apresentacao.tex
pdflatex apresentacao.tex
```

O ficheiro de referências é `Tese/refs.bib` (caminho `../refs` a partir da pasta da apresentação). No Cursor/VS Code com LaTeX Workshop, use a receita **«pdflatex ➞ bibtex ➞ pdflatex × 2»** para compilar com bibliografia.

## Figuras

Os slides usam caminhos relativos à pasta `tese_apresentacao/`:

- `../imgs/DAG_v2.png` — DAG (comportamento parlamentar e lobby). Se não existir, o slide mostra um placeholder.
- Outras figuras (resultados H1–H3) podem ser incluídas opcionalmente nos slides; os caminhos seriam `../figures/h1_test/`, `../figures/h2_test/`, `../figures/h3_test/`.

As pastas `imgs/` e `figures/` ficam em `Tese/`, ao lado de `tese_apresentacao/`.
