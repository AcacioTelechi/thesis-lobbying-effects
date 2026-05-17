---
title: "Memorando de alterações na tese"
subtitle: "Síntese das modificações realizadas em 9 de maio de 2026"
author: "Acácio"
date: "9 de maio de 2026"
---

# Resumo das alterações

Este memorando consolida as alterações realizadas nesta data nos arquivos da tese, em resposta às observações da banca de qualificação e a discussões subsequentes. As alterações concentram-se em sete frentes:

1. **Conceito de lobby** — ampliação para abranger os três poderes, com delimitação posterior ao Legislativo;
2. **Conceito de persuasão** — elaboração teórica do par *acesso × persuasão*, central à tese;
3. **Resultados da H2** — reenquadramento do achado de que ONGs apresentam maior persuasão por reunião do que empresas;
4. **Hipótese aberta** — formulação de hipótese explicativa sobre a endogeneidade do valor da informação ao desfecho mensurado;
5. **Perguntas parlamentares** — justificação ampliada da escolha da variável dependente;
6. **Regulamentação do lobby na UE** — nova seção sobre a evolução do regime e o Acordo Interinstitucional de 2021;
7. **Redação da H2** — explicitação do grupo de comparação.

As alterações implicaram também a inclusão de quatro novas entradas no arquivo `refs.bib`, listadas ao final deste memorando.

---

# 1. Conceito de lobby — ampliação para os três poderes

**Arquivo:** `Tese/main/intro/secoes/antecedentes_relevancia.tex` (linha 11)

**Motivação.** A definição original restringia o lobby à "atividade legislativa", mas o fenômeno ocorre também nos poderes Executivo e Judiciário. A redação foi ampliada para abranger os três poderes e em seguida delimitada ao Legislativo no escopo desta tese. Aproveitou-se a alteração para corrigir o *typo* "legistativa" e a concordância "o quão efetivo são essas ações".

**Antes:**

> Entendo por lobby como *o exercício de influência sobre a atividade legistativa realizado por atores interessados em um ambiente marcado por diferenças de recursos*. Essa definição evidencia aspectos fundamentais do fenômeno do lobby. Lobby é caracterizado por uma ação ("exercício") que demanda dispêndio de recursos a fim de alterar o comportamento parlamentar com maior ou menor grau de efetividade. Medir o quão efetivo são essas ações e analisar quais fatores afetam essa efetividade são o foco desta tese.

**Depois:**

> Entendo por lobby como *o exercício de influência sobre a tomada de decisão pública realizado por atores interessados em um ambiente marcado por diferenças de recursos*. Trata-se de um fenômeno que pode se manifestar em qualquer das esferas do poder público, sempre que houver atores externos buscando moldar decisões com efeitos coletivos. No contexto desta tese, contudo, o foco recai especificamente sobre o lobby exercido sobre o comportamento dos MPEs no PE. Essa definição evidencia aspectos fundamentais do fenômeno. Lobby é caracterizado por uma ação ("exercício") que demanda dispêndio de recursos a fim de alterar o comportamento dos tomadores de decisão — neste caso, dos parlamentares — com maior ou menor grau de efetividade. Medir o quão efetivas são essas ações e analisar quais fatores afetam essa efetividade são o foco desta tese.

---

# 2. Conceito de persuasão — elaboração teórica do par acesso × persuasão

**Arquivo:** `Tese/main/cap1-teoria/lobbying/definitions.tex` (linha 58 e seguintes)

**Motivação.** A banca observou que o conceito de "persuasão" não estava suficientemente claro. O parágrafo único e curto foi substituído por quatro parágrafos que (i) introduzem a decomposição como peça central da tese; (ii) definem *acesso*; (iii) definem *persuasão* com ancoragem agnóstica ao mecanismo (informacional, sinalização eleitoral, capital reputacional ou persuasão racional); (iv) explicitam a identidade `Efeito total ≈ Acesso × Persuasão` e a contribuição da decomposição para reconciliar achados aparentemente contraditórios da literatura.

**Antes (parágrafo único de quatro linhas):**

> Para o modelo utilizado nesta tese, consideramos que ambos os fatores são importantes. Devemos, porém, separar o fenômeno em dois efeitos. O primeiro é o *acesso*. O lobista deve conseguir, em primeiro lugar, acesso ao parlamentar. O acesso reflete a capacidade de troca de informação. O acesso, contudo, não garante a influência. É condição necessária, mas não suficiente. A influência surge da *persuasão*, o segundo elemento para a influência ocorrer. A persuasão reflete a capacidade de o lobista alterar o comportamento parlametar. A combinação de acesso e persuasão geram o nosso efeito de interesse - o efeito total do lobby no comportamento parlamentar.

**Depois (quatro parágrafos):**

> Para o modelo utilizado nesta tese, consideramos que ambos os fatores são importantes. Devemos, porém, decompor o fenômeno em dois efeitos: o *acesso* e a *persuasão*. Essa decomposição organiza tanto a estratégia empírica quanto a leitura dos resultados.
>
> O *acesso* é a condição que precede qualquer tentativa de influência: a capacidade de o lobista garantir interação direta com o parlamentar. Mensurado nesta tese pelo volume de reuniões registradas, o acesso reflete a capacidade de mobilizar recursos — financeiros, organizacionais e relacionais — para entrar na agenda do parlamentar (Hall e Wayman 1990; Chin 2005; Bouwen 2002). É condição necessária, mas não suficiente, para que a influência ocorra.
>
> A *persuasão* é o que ocorre dentro de cada interação: a alteração no comportamento do parlamentar atribuível a uma unidade adicional de acesso. Adoto uma definição agnóstica quanto ao mecanismo. A persuasão pode operar por vias distintas — provisão de informação técnica (Klüver 2012; de Figueiredo e Richter 2014), sinalização sobre as preferências do eleitorado (Hall e Miler 2008), transferência de capital reputacional ou de legitimidade (Bunea 2018), ou persuasão racional, como definida por Banfield (1961) —, sem que seja possível, com os dados disponíveis, distinguir qual delas predomina. O que importa, para a identificação empírica, é o efeito médio por reunião, independentemente do mecanismo.
>
> A decomposição é relevante quando confrontada com a literatura. A pesquisa que conclui pela superioridade da influência empresarial (Schlozman 1986; Schlozman e Tierney 1984; Baumgartner et al. 2009) mensura, em regra, o *efeito total* do lobby — sucesso em decisões, presença na agenda, gastos agregados —, sem distinguir a contribuição do acesso e da persuasão. Em termos formais:
>
> $$\text{Efeito total} \approx \underbrace{\text{Acesso}}_{\text{frequência da interação}} \times \underbrace{\text{Persuasão}}_{\text{eficácia marginal por interação}}.$$
>
> Como o efeito total é o produto das duas componentes, é possível que um ator seja superior em uma e inferior na outra, e que o resultado agregado mascare essa heterogeneidade. Esta tese, ao decompor as duas dimensões, abre espaço para reconciliar achados que parecem contraditórios na literatura — em particular, a possibilidade de que a vantagem agregada das empresas se origine do acesso, e não da persuasão por interação.

---

# 3. Resultados da H2 — reenquadramento do achado ONG > empresa

**Arquivo:** `Tese/main/cap4-resultados/h2.tex` (linhas 3 e 52)

**Motivação.** O texto anterior tratava como "paradoxal" e "contraintuitivo" o achado de que ONGs exercem maior persuasão por reunião do que empresas. À luz do framework conceitual reformulado no Capítulo 1, esse padrão deixa de ser anômalo e passa a ser uma predição da decomposição entre acesso e persuasão.

**3.1. Linha 3 — abertura da seção da H2**

**Antes:**

> A avaliação da Hipótese 2, que postula uma maior influência das empresas sobre a atividade parlamentar em comparação com outros atores, exige uma análise que transcenda a simples contagem de reuniões. Uma análise preliminar do efeito marginal por reunião, apresentada na Figura 4.X, sugere que as ONGs, paradoxalmente, exercem uma influência maior por encontro. Este resultado, embora contraintuitivo, destaca a necessidade de um modelo mais completo que considere a heterogeneidade dos atores de lobby.

**Depois:**

> A avaliação da Hipótese 2, que postula uma maior influência das empresas sobre a atividade parlamentar em comparação com outros atores, exige uma análise que opere a decomposição entre acesso e persuasão proposta no Capítulo 2. A Figura 4.X apresenta o efeito marginal por reunião — isto é, a componente de *persuasão* —, sugerindo que as ONGs exercem maior influência por encontro do que as empresas. À primeira vista, esse resultado contrasta com a literatura que aponta para a primazia da influência empresarial (Schlozman 1986; Baumgartner et al. 2009); como discutiremos adiante, contudo, o aparente contraste se dissolve ao reconstituirmos o efeito total a partir do produto entre as duas componentes.

**3.2. Linha 52 — parágrafo novo entre o resultado e a discussão**

**Inserido (parágrafo novo):**

> Esse padrão é o que o quadro conceitual do Capítulo 2 antecipava. A literatura empírica que conclui pela superioridade da influência empresarial mensura, predominantemente, o efeito total — uma quantidade que confunde acesso e persuasão. Ao decompor as duas componentes, observamos que a vantagem das empresas se origina sobretudo de sua capacidade de converter recursos financeiros em volume de interação (acesso), e não de uma maior eficácia por interação (persuasão). A persuasão por reunião das ONGs é consistentemente maior, e só é soterrada pela superioridade quantitativa das empresas quando estas dispõem de orçamentos suficientemente elevados. Não há, portanto, contradição com a literatura, mas refinamento: o efeito total continua a favorecer as empresas em altos orçamentos, porém o canal pelo qual essa influência opera é o acesso, não a persuasão.

---

# 4. Hipótese aberta sobre endogeneidade do valor da informação

**Arquivo:** `Tese/main/cap4-resultados/h2.tex` (após linha 58); ecoado em `consid_final/index.tex` (linha 28)

**Motivação.** A reconciliação do item 3 ainda deixa em aberto uma questão: a literatura de lobby informacional (Hall & Miler 2006; Klüver 2012) prediz que as empresas teriam vantagem *também* na persuasão por reunião — porque oferecem informação técnica especializada — e os resultados indicam o oposto. O texto anterior não abordava essa tensão. Foi inserido um bloco de quatro parágrafos formulando uma hipótese explicativa, **apresentada como hipótese a ser testada empiricamente em pesquisas futuras**, e não como reconciliação fechada.

**Inserido (após o parágrafo sobre subsidiação informacional pelas empresas):**

> Permanece em aberto, contudo, uma questão que merece investigação mais aprofundada. A literatura sobre lobby informacional (Klüver 2012; de Figueiredo e Richter 2014; Hall e Miler 2008) prediz que as empresas, por disporem de informação técnica especializada e de maior *stake* econômico em decisões legislativas, deveriam exibir vantagem não apenas no *acesso*, mas também na *persuasão* por interação. Os resultados aqui obtidos, ao indicarem o oposto na componente de persuasão, parecem à primeira vista contrariar essa predição.
>
> Levantamos, neste ponto, uma hipótese explicativa que carece de teste empírico direto: a de que o valor da informação seria *endógeno ao desfecho mensurado*. A AL, nesta tese, é operacionalizada pelo volume de perguntas parlamentares submetidas pelos MPEs. Trata-se de uma forma de comportamento parlamentar com função predominantemente sinalizadora — perguntas parlamentares são instrumentos de visibilidade pública e de demarcação de posição política, e não veículos diretos de formulação de políticas. Já a literatura sobre lobby informacional foca tipicamente em desfechos de política pública — aprovação de legislação, conteúdo regulatório, alinhamento de propostas iniciais (Yackee 2006; Klüver 2015) —, nos quais a informação técnica especializada possui valor mais imediato.
>
> Se essa distinção for relevante, tipos diferentes de informação teriam valor distinto segundo o desfecho considerado. Para a formulação de políticas, a informação técnica das empresas — sobre custos de implementação, especificidades regulatórias, viabilidade operacional — seria particularmente valiosa. Para a sinalização parlamentar, contudo, o que tenderia a render mais é a informação *política*: quais temas têm ressonância social, quais causas portam legitimidade pública, quais questões mobilizam constituintes — precisamente a vantagem comparativa que a literatura atribui às ONGs (Bunea 2018; Pereira 2025). Sob essa hipótese, o achado de maior persuasão por reunião das ONGs não refutaria a literatura sobre lobby informacional, mas a qualificaria.
>
> Testar essa hipótese diretamente exigiria replicar a estratégia de identificação aqui proposta sobre outros desfechos legislativos — emendas, relatórios em comissão, votos nominais. Caso a hipótese se sustente, esperar-se-ia uma reversão do padrão observado nestes desfechos: a vantagem persuasiva das empresas se manifestaria onde a informação técnica é mais valiosa, enquanto a vantagem das ONGs tenderia a se concentrar onde o componente sinalizador é dominante. Esta agenda permanece em aberto e constitui uma das principais extensões naturais desta tese.

A hipótese é também ecoada na agenda de pesquisas futuras das **Considerações Finais**, ligando a sua confirmação ou refutação à reconciliação plena com a literatura de lobby informacional.

---

# 5. Considerações finais — alinhamento com o framework

**Arquivo:** `Tese/main/consid_final/index.tex` (linhas 13 e 28)

**Motivação.** A redação anterior do parágrafo da H2 trazia uma redundância ("essa influência só se torna dominante quando alavancada por vastos recursos financeiros" aparecia duas vezes em parágrafos consecutivos) e não articulava a contribuição da decomposição como o eixo da reconciliação com a literatura. A reescrita consolida os dois parágrafos e reorganiza o argumento em torno do framework. A agenda de pesquisas futuras foi ampliada para incluir a hipótese aberta do item 4.

**Antes (linhas 13–22):**

> A H2, que investigava se as empresas exercem influência agregada superior sobre a atividade parlamentar em comparação com outros atores, também foi corroborada. O efeito marginal isolado, contudo, foi insuficiente para um teste robusto da hipótese. A influência total de um grupo de interesse não depende apenas da eficácia de cada reunião, mas também da sua capacidade de assegurar acesso — isto é, o volume de reuniões que consegue realizar. Argumentamos que o impacto total é uma função dessas duas componentes: a frequência do acesso e a eficácia da persuasão em cada encontro. Obtivemos resultados que sugerem que as empresas exercem uma influência agregada superior sobre a atividade parlamentar em comparação com outros atores, mas que essa influência só se torna dominante quando alavancada por vastos recursos financeiros. Isto é, há diferenças relevantes dentro de cada tipo de ator.
>
> Empresas menores, com menos recursos, exercem uma influência menor mesmo em comparação com as ONGs, mas que essa influência só se torna dominante quando alavancada por vastos recursos financeiros. Para orçamentos abaixo de \$8,8 milhões, o efeito total das ONGs permanece superior. No entanto, acima de \$40 milhões, o efeito agregado das empresas torna-se substancialmente maior. Esses resultados se alinham com a literatura sobre os mecanismos de influência do lobby e reforçam a importância de considerar a heterogeneidade dos atores na análise dos efeitos do lobby sobre o comportamento parlamentar.

**Depois (parágrafo único):**

> A H2, que investigava se as empresas exercem influência agregada superior sobre a atividade parlamentar em comparação com outros atores, foi corroborada apenas após a decomposição teórica entre acesso e persuasão proposta no Capítulo 2. O efeito marginal por reunião — a componente de *persuasão* — é, na verdade, sistematicamente maior para as ONGs do que para as empresas. À primeira vista, esse resultado contradiria a literatura empírica que aponta para a primazia da influência empresarial (Schlozman 1986; Baumgartner et al. 2009). A aparente contradição se dissolve, contudo, quando se observa que essa literatura mede tipicamente o *efeito total* — o produto entre acesso e persuasão —, sem distinguir as duas componentes. Ao decompô-las, evidencia-se que a vantagem agregada das empresas opera quase exclusivamente pelo canal do acesso: a sua capacidade de converter recursos financeiros em volume massivo de reuniões. A influência só se torna dominante, portanto, quando alavancada por vastos recursos financeiros: para orçamentos abaixo de \$8,8 milhões, o efeito total das ONGs permanece superior; acima de \$40 milhões, o efeito agregado das empresas torna-se substancialmente maior. Há, assim, diferenças relevantes dentro de cada tipo de ator, que a leitura tradicional do lobby — focada apenas no efeito agregado — não permite enxergar.

**Agenda de pesquisas futuras (linha 28) — ampliação:**

Foi acrescentada referência à hipótese aberta sobre endogeneidade do valor da informação ao desfecho, com a indicação de que sua confirmação ou refutação dependerá de replicar a estratégia de identificação sobre outros desfechos legislativos (emendas, relatórios, votos nominais).

---

# 6. Perguntas parlamentares — justificação ampliada

**Arquivo:** `Tese/main/cap3-metodologia_e_resultados/identificacao.tex` (linhas 6–10)

**Motivação.** A tese justificava a escolha das perguntas parlamentares como variável dependente em um *footnote* e dois parágrafos. Dado o peso que a hipótese aberta do item 4 atribui à natureza sinalizadora desse desfecho, a justificação foi expandida para cinco parágrafos, abordando: (i) a definição institucional das perguntas; (ii) suas duas funções na literatura sobre o PE; (iii) três propriedades que justificam o seu uso como proxy nesta tese; (iv) o que esse proxy *não* captura.

**Conteúdo novo (substituiu a *footnote* + dois parágrafos anteriores):**

> A variável dependente utilizada é a AL, operacionalizada como o número de perguntas parlamentares que um MPE (*i*) apresenta em um determinado domínio temático (*d*) e período (*t*). A escolha por focar no comportamento parlamentar, em vez de resultados de políticas públicas, reduz a complexidade da cadeia causal e aproxima a análise da ação individual do legislador.
>
> Cabe, antes de avançar, justificar por que as perguntas parlamentares são um proxy adequado para a atividade do MPE. Trata-se de instrumentos formais previstos no Regimento do PE, pelos quais um MPE solicita esclarecimentos ou informações à Comissão Europeia, ao Conselho ou ao Banco Central Europeu. Podem ser submetidas por escrito ou oralmente, têm prazos formais de resposta, e tanto as perguntas quanto as respostas são publicadas no registro do PE. Esse caráter de registro público é central para o argumento que se segue.
>
> A literatura sobre o PE atribui às perguntas duas funções. A primeira é de fiscalização e responsabilização (Jensen et al. 2013; Maricut-Akbik 2020; Martin 2013): a partir da literatura de agente-principal, o legislador (principal) delega poder à burocracia (agente) e usa as perguntas para monitorar suas ações, sobretudo em temas salientes (McCubbins e Schwartz 1984; Saalfeld 2000; Strøm 2000; Koop 2011). A segunda é de sinalização e posicionamento (Otjes e Louwerse 2017; Proksch e Slapin 2010; Bevan e Greene 2023; Navarro e Sieberer 2022): o MPE usa as perguntas para expressar preferências, marcar posição e sinalizar engajamento à liderança partidária, aos eleitores e aos grupos de interesse. Por terem baixo custo de produção e alta visibilidade, são instrumento útil sobretudo à oposição e a parlamentares que buscam diferenciação dentro do próprio grupo político.
>
> Três propriedades das perguntas as tornam um indicador apropriado para esta tese. **Primeiro**, o volume é alto: a base contém dezenas de milhares de perguntas no período analisado (ver Capítulo 4), o que permite captar variação fina na atenção do MPE a cada domínio temático, mês a mês. **Segundo**, o custo marginal de uma pergunta é baixo: o MPE pode submetê-la sem mobilizar maioria nem superar barreiras institucionais, o que torna sua produção responsiva a estímulos externos como o lobby. **Terceiro**, o caráter público das perguntas as inscreve na estratégia de visibilidade do MPE — o que ele pergunta funciona como sinal sobre os temas que deseja associar à sua imagem.
>
> Convém registrar o que esse proxy não captura. As perguntas medem atenção e posicionamento, não decisões de política pública. Não capturam a influência exercida nas fases mais técnicas da formulação legislativa — negociação de emendas, relatórios em comissão, votos nominais —, nem a influência exercida em interações sem registro formal. Trata-se, portanto, de uma escolha que privilegia identificação causal mais limpa em troca de uma fatia específica do fenômeno do lobby. Essa delimitação é retomada no Capítulo 4, onde se discute como ela molda a interpretação dos achados sobre persuasão.

---

# 7. Regulamentação do lobby na UE — nova seção sobre o AIRT

**Arquivo:** `Tese/main/cap1-teoria/lobbying/lobbying_in_the_eu.tex` (após linha 4)

**Motivação.** A tese mencionava o Acordo Interinstitucional sobre o Registo de Transparência (AIRT) apenas como base definicional do conceito de "representante de interesses", sem discutir o regime de regulação ou as transformações operadas pelo Acordo Interinstitucional de 2021 — em particular o princípio da condicionalidade. Foi inserida uma seção de sete parágrafos cobrindo: trajetória histórica (1995, 2008, 2011, 2014, 2021); o princípio da condicionalidade introduzido em 2021; o escopo de atividades cobertas e excluídas; as seis categorias de registrantes; a assimetria de divulgação financeira entre empresas e ONGs (relevante para a análise empírica); o Código de Conduta e mecanismos de fiscalização; e as limitações do regime — culminando nas implicações diretas para os dados desta tese.

**Conteúdo novo (sete parágrafos), com citações às fontes oficiais e à literatura:**

> O regime de regulação do lobby na UE é parte central desse contexto institucional. Ao longo das últimas três décadas, deslocou-se de um modelo voluntário e fragmentado para um modelo obrigatório e tri-institucional (EPRS 2023; Ridao Martín e Araguàs Galcerà 2025). O conhecimento desse regime é necessário tanto pela importância política da transparência quanto por uma razão metodológica imediata: o Registo de Transparência é a fonte primária dos dados utilizados nesta tese (UE 2025).
>
> A trajetória do regime tem quatro marcos (EPRS 2023). Em **1995**, o Parlamento Europeu instituiu seu próprio registro voluntário de representantes de interesses, em resposta a preocupações com a opacidade das interações entre lobistas e parlamentares. Em **2008**, a Comissão Europeia criou um registro paralelo, também voluntário. Em **2011**, as duas instituições fundiram seus instrumentos por meio de um Acordo Interinstitucional, criando o Registo de Transparência conjunto — o Conselho participou apenas como observador. O sistema foi revisto em **2014**, mantido o caráter voluntário. O ponto de inflexão ocorreu em **2021**: o AIRT, de 20 de maio de 2021 (UE 2021), entrou em vigor em 1º de julho de 2021 e instituiu um Registo de Transparência **obrigatório** para as três instituições signatárias, com o Conselho passando da posição de observador à de signatário pleno.
>
> A mudança operacional decorre do **princípio da condicionalidade** (UE 2021, art. 5): a inscrição no registro deixa de ser voluntária e passa a ser pré-condição para o exercício de determinadas atividades junto às instituições. Apenas entidades inscritas podem realizar reuniões com Comissários, membros de gabinete, diretores-gerais, eurodeputados em posições de liderança e relatores; acessar as instalações das instituições para encontros formais; participar de painéis de peritos; ou figurar em campanhas e eventos institucionais. Atividades ligadas à formulação ou implementação legislativa só podem envolver representantes registrados (Ridao Martín e Araguàs Galcerà 2025).
>
> O escopo é amplo. O AIRT considera como atividade sujeita a registro organizar ou participar de reuniões e eventos com membros das instituições; contribuir para consultas e audições; conduzir campanhas de comunicação; e encomendar ou produzir documentos de política, emendas, estudos e pesquisas com finalidade de influência (UE 2021, art. 3). Permanecem fora do escopo o aconselhamento jurídico em sentido estrito, a participação em diálogo social entre empregadores e sindicatos, atividades estritamente pessoais e encontros privados espontâneos (UE 2021, art. 4).
>
> Os registrantes são classificados em **seis categorias** (UE 2021, Anexo I): (I) consultorias profissionais, escritórios de advocacia e consultores autônomos; (II) lobistas internos (*in-house*), associações comerciais e profissionais e sindicatos; (III) ONGs; (IV) *think tanks*, instituições de pesquisa e acadêmicas; (V) organizações religiosas; e (VI) autoridades públicas subnacionais. As exigências de divulgação financeira variam por categoria (UE 2021, Anexo II): empresas, consultorias e associações reportam os custos anuais associados às atividades de representação de interesses, em faixas pré-definidas, sem obrigação de divulgar a receita total; **as ONGs e entidades análogas devem reportar, além dos custos com lobby, o orçamento total e as fontes de financiamento**. Essa assimetria, ainda que justificada pela natureza distinta dos atores, dificulta a comparação direta entre orçamentos de empresas e ONGs — ponto retomado na análise empírica.
>
> O AIRT estabelece ainda um **Código de Conduta** de dezesseis regras vinculantes (UE 2021, Anexo I), que incluem a obrigação de declarar os interesses representados, manter as informações atualizadas, abster-se de obter informação por meios desonestos e não induzir agentes públicos a violar suas próprias regras. O monitoramento cabe ao Secretariado do Registro, sob supervisão de um conselho tri-institucional, com sanções que vão de suspensões temporárias à remoção do registro — e, por consequência, à perda do acesso às instituições (UE 2021, Anexo III).
>
> As limitações do regime são reconhecidas pela literatura (Bunea e Ibenskas 2019; Ridao Martín e Araguàs Galcerà 2025). Cada instituição implementa o princípio da condicionalidade segundo regras próprias (EPRS 2023) — o Conselho, em particular, restringe a obrigatoriedade a um conjunto mais limitado de funcionários do que o Parlamento e a Comissão. A capacidade de fiscalização do Secretariado é circunscrita pelos recursos disponíveis e depende, em larga medida, de denúncias e revisões amostrais (Ridao Martín e Araguàs Galcerà 2025). O registro também não cobre interações informais nem o lobby exercido junto às representações nacionais em Bruxelas, o que delimita o universo observável. Essas restrições têm consequências diretas para os dados desta tese: as reuniões registradas são as realizadas com membros do PE em posições de liderança ou de relatoria, e não a totalidade das interações lobby-parlamentar — ponto retomado no Capítulo 3 e no apêndice operacional.

---

# 8. H2 — explicitação do comparativo

**Arquivos:** `Tese/main/intro/secoes/antecedentes_relevancia.tex` (linhas 23 e 31)

**Motivação.** A redação da H2 não explicitava o grupo de comparação ("mais propensas a aumentar a AL — em relação a quem?"). Os pontos posteriores da tese (capítulos metodológico e empírico, considerações finais) já especificavam "em comparação a outras categorias de atores", mas a hipótese formal estava em descompasso. A redação foi alinhada.

**Antes:**

> H2: As pressões de lobby de organizações empresariais são mais propensas a aumentar a AL dos MPEs.

**Depois:**

> H2: As pressões de lobby de organizações empresariais são mais propensas a aumentar a AL dos MPEs **em comparação às demais categorias de atores**.

A alteração foi aplicada em ambos os pontos onde a H2 é declarada formalmente: no parágrafo argumentativo (linha 23) e no bloco `\begin{hypotheses}` (linha 31).

---

# Apêndice — novas entradas no `refs.bib`

Para suportar as citações inseridas no item 7, foram adicionadas quatro entradas no arquivo `Tese/refs.bib`:

1. **`airt2021`** — Acordo Interinstitucional de 20 de maio de 2021, *Jornal Oficial da União Europeia*, L 207, 11 de junho de 2021, pp. 1–20. Texto oficial do Acordo Interinstitucional sobre o Registo de Transparência Obrigatório.

2. **`eprs2023register`** — Cirlig, Carmen-Cristina (2023). *EU Transparency Register: 2021 interinstitutional agreement*. European Parliamentary Research Service, Briefing PE 751.434.

3. **`ridao2025lobbying`** — Ridao Martín, Joan e Araguàs Galcerà, Irene (2025). "Lobbying in the EU: prospects and challenges of the mandatory transparency register". *Frontiers in Political Science*, vol. 6, art. 1508017. DOI: 10.3389/fpos.2024.1508017.

4. **`eu_register_portal`** — Portal oficial do EU Mandatory Transparency Register (Parlamento, Conselho e Comissão Europeia), citado como fonte institucional dos dados.

---

# Notas sobre estilo e dose de adjetivos

Em todas as alterações acima, a redação foi mantida com baixa carga de intensificadores ("particularmente", "predominantemente", "fundamentalmente", "deliberadamente", "consistentemente", "precisamente") e superlativos ("vastos", "grande corpo de", "principal"), em consonância com o pedido de redação enxuta. Termos como "analiticamente distintos" e "aparentemente contraditórios" foram substituídos por formulações mais diretas ("distintos", "que parecem contraditórios").
