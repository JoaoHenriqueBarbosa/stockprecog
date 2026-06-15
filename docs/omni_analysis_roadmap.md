# Análise Omni — stockprecog: estado, taxa de ganho e roadmap até SOTA 2024+

_Gerado por workflow multi-agente (12 agentes: revisão código+método, trajetória, 4 frentes de pesquisa SOTA, síntese, 3 verificadores adversariais, finalização)._


## 1. Taxa de ganho por lever

| Lever | AUC | ΔAUC | Sharpe | DSR | Veredito |
|---|---|---|---|---|---|
| L0 baseline | 0.5125 | — | — | —(split único) | ponto de partida cego economicamente |
| L1 brapi limpo | 0.5125 | 0.0000 | — | 0 | dados NÃO eram o gargalo; de-risk survivorship/bonif |
| L2 CPCV+DSR | 0.5146 | +0.0021* | −0.022 | 0 | revelou que não há edge (régua) |
| L3 fracdiff | 0.5232 | +0.0086 | +0.028 | 0 | único salto real de AUC (~75% do ganho) |
| L4 meta-label | 0.5224 | −0.0008 | −0.086 | 0 | 1º retorno marginal NEGATIVO |

_*L2 mudou a régua, não a feature — ΔAUC não comparável._


### Diagnóstico
Há duas séries sendo tratadas como uma. AUC subiu de 0.5125 → 0.5232 (edge sobre coin-flip quase dobrou, +0.0125 → +0.0232) — isso é real e atribuível quase inteiramente a UM lever: fracdiff (L3, +0.0086, ~75% de todo o ganho de AUC pós-baseline). Os outros levers de AUC renderam ~0 (L1) ou são mudança de régua, não de feature (L2). Mas o edge ECONÔMICO (DSR/PSR) está cravado em 0 nos cinco levers, sem exceção. Não é coincidência nem 'falta de mais features': DSR=0 com Sharpe oscilando em torno de zero é o resultado matematicamente esperado quando você deflaciona por nº de trials (CPCV gera muitos paths), skew/curtose e comprimento de track curto. O AUC de 0.52 é discriminação de ranking marginal que a construção de estratégia (long-short por mediana, sem custos, sem sizing) destrói antes de virar P&L. E a própria capacidade de MEDIR edge está comprometida: labels de 10 dias amostrados diariamente inflam t-stat ~√10, então tanto Sharpes positivos quanto negativos estão num intervalo de confiança que cruza zero. Você não tem um problema de 'edge baixo' — tem um problema de 'edge não-mensurável e ranking que não converte em P&L'.

### O que realmente move o edge
Nada moveu edge econômico ainda — DSR=0 é o estado real, não pessimismo. Em AUC, só fracdiff (L3) moveu de verdade (~75% do ganho pós-baseline); o resto foi infraestrutura (L1), honestidade de medição (L2) ou regressão (L4). Para mover EDGE (distinto de AUC), em ordem de alavancagem: (1) PRÉ-REQUISITO — sample uniqueness / sequential bootstrap / retornos não-sobrepostos (LdP cap.4). Sem isso o t-stat infla ~√10 e você não consegue nem MEDIR edge honestamente; o DSR=0 é em parte porque o N efetivo é ~10× menor que o nominal. Isto não adiciona edge, mas torna o edge mensurável — tudo depois disso é cego sem ele. (2) Tradução ranking→P&L: bet sizing (proba→tamanho) + modelo de custo + execução realista. O long-short por mediana joga fora a informação de ranking que o AUC 0.523 captura — é onde o sinal está vazando. (3) Eixo de feature ORTOGONAL: a família price-derived (TA+fracdiff) está exausta. Edge novo precisa de dados que NÃO são preço — fundamentos point-in-time, fluxo de ordens/microestrutura, cross-asset. (4) Corrigir survivorship (universo point-in-time, não IBrX de hoje) — atualmente você pode estar medindo edge inflado por sobreviventes. DL/GPU fica para depois de (2)+(3): só vale a pena extrair sinal com rede neural quando existe sinal e amostra que justifiquem; em 28 tickers de TA, é overfit garantido.

## 2. Bugs metodológicos (auditoria AFML)

- **[critico]** `cpcv.py:78-92 + evaluate.py:91 (deflated_sharpe)` — sr_star usa var(sharpes dos 15 splits do CPCV) como variancia de tentativas de research e n_trials=C(N,k)=15. Paths CPCV sao reamostragens de UMA config (variancia amostral do estimador), nao trials independentes de multiple-testing; alem disso sao correlacionados (treinos compartilham dados) e n_trials e um botao de CV, nao esforco de pesquisa. sr_star vira limiar sem significado; DSR=0 e 'certo pelo motivo errado' e nao detectaria skill se houvesse.
  - _fix:_ Manter um log de trials de pesquisa (Levers 0-4 + hyperparams descartados); deflated_sharpe(sr, sr_trials=<sharpes desses configs>, n_trials=<count real>). Usar a variancia do CPCV apenas como dispersao do estimador, nunca como deflacao.
- **[critico]** `evaluate.py:87,91 (pooled / n do PSR-DSR)` — n=len(pooled) onde pooled concatena retornos de teste dos 15 splits. Cada data pertence a C(N-1,k-1)=5 combinacoes de teste -> cada retorno duplicado ~5x; somado a retornos sobrepostos de 10 dias (autocorrelacao), o fator sqrt(n-1) infla a confianca do PSR/DSR maciçamente. skew/kurt do pooled tambem viesados pela duplicacao.
  - _fix:_ Deduplicar pooled por (ticker,date); usar n = n_datas_unicas/HORIZON * avg_uniqueness. Reportar SR overlap-robusto.
- **[alto]** `labeling.py:54 (fwd_ret)` — fwd_ret usa close[end] (barreira vertical) enquanto label e t1 resolvem em res (toque, ex t+3). O Sharpe/DSR mede um sinal diferente do rotulado; e o info-horizon real da amostra (end) excede o horizonte de purge (t1=res), tornando o purge curto demais para o retorno usado no backtest.
  - _fix:_ fwd_ret = close[res]/close[t]-1 (path-consistente com label e t1). Se a intencao for segurar sempre H dias, entao t1=dates[end].
- **[alto]** `evaluate.py:73 / meta_evaluate.py:70,78 + ausencia global (cap.4)` — Nenhum .fit passa sample_weight. Labels sobrepostos (ate 10d, amostrados diariamente) geram amostras redundantes no treino (effective N << N, overfit a regimes lotados) e retornos autocorrelacionados na avaliacao -> infla AUC-estabilidade, t-stat (~sqrt(10)) e PSR. meta_evaluate.py:105 calcula t=mean/(std/sqrt(15)) tratando 15 splits correlacionados como iid -> t-stat invalido.
  - _fix:_ Average uniqueness (cap.4) -> sample_weight no LGBM; sequential bootstrap; deflacionar n por uniqueness na inferencia de SR; reportar variante com amostragem nao-sobreposta (stride=HORIZON).
- **[alto]** `config.py:17-31 (UNIVERSE)` — Universo = IBrX liquido de HOJE, nao point-in-time; ativos deslistados/quebrados ausentes; lista UNRELIABLE adiciona selecao. Enviesa base-rate dos labels e retornos para cima. Documentado mas nao resolvido.
  - _fix:_ Composicao historica do indice ponto-no-tempo (membership por data); incluir deslistados ate a data de saida.
- **[medio]** `features.py:43 (fd_vol21)` — g['vol_21'].bfill() antes do fracdiff: bfill preenche NaN do warm-up com o primeiro valor FUTURO valido -> lookahead nas linhas iniciais, que alimentam os primeiros folds do CPCV.
  - _fix:_ Usar ffill() ou deixar NaN (fracdiff ja guarda contra NaN). Nunca bfill em serie temporal.
- **[medio]** `cpcv.py:45 + config.py:43 (embargo)` — EMBARGO_DAYS=HORIZON=10 aplicado como timedelta(10,'D')=10 dias-calendario, mas HORIZON sao 10 dias-uteis (~14 calendario). Embargo subdimensionado -> vaza apos o bloco de teste.
  - _fix:_ Embargo por posicao no indice de pregoes (trading days), ou ceil(HORIZON*7/5)+folga em calendario.
- **[medio]** `meta_evaluate.py:65-66 (split interno)` — Primario treina em date<=cut cujos labels resolvem ~10d depois, dentro do conjunto meta (date>cut) -> vaza primario-treino para meta-treino. Sem purge no boundary cut.
  - _fix:_ Descartar linhas do primario com t1 > cut antes de gerar meta-labels.
- **[baixo]** `labeling.py:43-44 (empate na mesma barra)` — hit_up & hit_dn no mesmo bar -> label 0 forcado. Com PT=SL, 'SL primeiro' e arbitrario e enviesa base-rate para fracasso, descartando ambiguidade genuina.
  - _fix:_ Regra por proximidade do open, marcar NaN/dropar, ou usar dado intraday.
- **[baixo]** `fracdiff.py:43-58 + features.py:11 (d=0.4 global)` — min_ffd_d roda ADF na serie inteira (inclui periodo de teste) para escolher d -> lookahead no escalar d. Leve.
  - _fix:_ Escolher d em janela de warm-up purgada, ou documentar como desprezivel.
- **[baixo]** `evaluate.py:66-105 (paths vs splits)` — '5 paths' e nominal: o codigo tira media de 15 splits e nunca reconstroi os 5 paths contiguos de backtest. Cosmetico, mas e onde a contagem de trials do DSR se confunde.
  - _fix:_ Reconstruir os C(N,k)*k/N paths contiguos para a verdadeira distribuicao de backtest (cap.12), separada da contagem de trials do DSR.

## 3. Roadmap final — 20 passos até warp drive (SOTA 2024+)

**Princípio:** rigor ANTES de edge. A régua tem que estar à prova de bala (1–7) antes de medir; dados honestos (8–12); economia real (13–16); só então modelo/features de fronteira (17–20). DL é o último passo, não o atalho.


### Fase A — Consertar a régua (1–7)

**1. Corrigir fwd_ret para a barreira efetiva (res) com preço de saída EOD-executável e tratar empate na mesma barra** _(esforço baixo)_
- hipótese: Três defeitos acoplados em labeling/backtest: (a) fwd_ret usa close[end] (barreira vertical t+10) enquanto label/purge resolvem em res (toque, ex t+3), medindo sinal diferente do rotulado e encurtando o purge; (b) assumir saída no NÍVEL da barreira (PT/SL·σ) com dados EOD é lookahead de execução — só se transaciona no close; (c) hit_up & hit_dn na mesma barra força label 0 ('SL primeiro' arbitrário com PT=SL), enviesando a base-rate para fracasso e contaminando a calibração da proba que sizing/ranking consomem. Corrigir os três é pré-condição para qualquer número econômico confiável.
- base SOTA: AFML cap.3 — triple-barrier path-consistente (retorno casa com o evento que rotulou e com o t1 do purge); realismo de execução EOD (saída em close[res] ou próximo close, não no nível da barreira).
- lift esperado: Nenhum lift de edge; corrige viés de sinal e de execução. Provável mudança material (e possível inversão de sinal) no Sharpe/meta; a saída executável pode reduzir o Sharpe MAIS que a própria correção res-vs-end.
- risco: Baixo. Risco real é descobrir Sharpe pós-correção pior — mas é a verdade, não regressão. Re-medir base-rate antes/depois do fix de empate para quantificar o viés.

**2. Eliminar lookahead do fd_vol21 (bfill→ffill/NaN)** _(esforço baixo)_
- hipótese: g['vol_21'].bfill() antes da fracdiff preenche o warm-up com o primeiro valor FUTURO, injetando lookahead exatamente nas linhas que alimentam os primeiros folds do CPCV — viola a invariante point-in-time central do projeto.
- base SOTA: AFML cap.5 (fracdiff causal) + princípio point-in-time: nunca bfill em série temporal.
- lift esperado: Nenhum lift; remove inflação otimista leve do AUC nos folds iniciais. fracdiff já guarda contra NaN.
- risco: Trivial.

**3. Embargo em dias-de-pregão, não dias-calendário** _(esforço baixo)_
- hipótese: EMBARGO=HORIZON=10 aplicado como timedelta(10 dias-calendário), mas HORIZON são 10 dias-úteis (~14 calendário). Embargo subdimensionado vaza após o bloco de teste.
- base SOTA: AFML cap.7 — embargo deve cobrir o horizonte de informação real; medir em posição no índice de pregões.
- lift esperado: Nenhum lift; fecha um canal de leak que infla AUC/DSR.
- risco: Baixo.

**4. d* da fracdiff walk-forward (remover lookahead do escalar na feature headline)** _(esforço medio · dep [3])_
- hipótese: d=0.4 é global e escolhido via ADF na série inteira (inclui teste) → lookahead no escalar da MAIOR feature do projeto (fd_close, +0.0086 desde Lever 3). Estimar d* em janela de warm-up purgada por fold (min_ffd_d já existe, nunca é chamado) fecha o último resíduo de lookahead. Pertence à fase da régua, não à caça ao edge: nenhum claim econômico é defensável enquanto a feature headline estiver vazada.
- base SOTA: AFML cap.5 — min_ffd_d via ADF sem lookahead de série inteira.
- lift esperado: Lift de AUC marginal ou nulo; valor é fechar o leak e tornar a maior feature defensável sob escrutínio. Risco de auto-tuning: se mal-purgado por fold, transforma leak leve em leak material.
- risco: Baixo, condicionado a purga correta do warm-up por fold.

**5. Purga no boundary cut do meta-labeling ou aposentadoria explícita do L4** _(esforço baixo)_
- hipótese: A auditoria flagou leak no boundary do meta-labeling (meta_evaluate.py:65-66): primário treina em date<=cut com labels que resolvem ~10d DENTRO do conjunto meta, sem purge no cut. O L4 já rodou com esse vazamento. Deixar o bug órfão é risco silencioso: se meta-labeling voltar ou se a lógica de split for reaproveitada, o leak ressurge.
- base SOTA: AFML cap.3/7 — purge two-sided no boundary entre treino primário e conjunto meta.
- lift esperado: Nenhum lift; remove falso-positivo histórico do L4 e de-risca reuso futuro.
- risco: Baixo. Decisão binária: descartar linhas do primário com t1>cut, OU declarar L4 morto e remover meta_evaluate.py do caminho quente.

**6. Suíte de testes de regressão do caminho quente** _(esforço medio · dep [1, 2, 3, 4, 5])_
- hipótese: Zero testes hoje: toda correção depende de inspeção manual. Travar o comportamento correto (alinhamento label↔t1↔fwd_ret, saída executável, empate, purge two-sided, d* purgado, ausência de linha primária com t1>cut no meta, pesos FFD, fórmulas PSR/DSR) impede que os 14 passos seguintes reintroduzam silenciosamente os bugs 1-5.
- base SOTA: Engenharia de pesquisa reprodutível; AFML insiste em validar a régua antes de confiar em resultados.
- lift esperado: Nenhum lift direto; de-risca todos os passos seguintes e converte os fixes 1-5 em invariantes verificadas.
- risco: Baixo. Custo é tempo, não risco.

**7. Selar lockbox OOS + pré-registrar stopping rule e hurdle de aceite** _(esforço baixo · dep [6])_
- hipótese: CPCV reusa todos os dados em todos os passos; ao chegar no DL o mesmo painel foi minerado ~20× em decisões SEQUENCIAIS e ADAPTATIVAS (garden of forking paths). O log de trials deflaciona o Sharpe assumindo trials quase-independentes — mas estes são correlacionados e adaptativos, então a deflação SUBESTIMA o esforço de busca. Sem holdout intocado e sem critério de parada definido a priori, o programa só para quando algo 'passa' = falso-positivo por construção.
- base SOTA: Bailey & López de Prado 2014 (deflated Sharpe pressupõe trials ~independentes); prática de pré-registro / holdout selado contra overfitting do PROGRAMA.
- lift esperado: Nenhum lift; é a única defesa contra o data-mining acumulado dos 20 passos. Define: (a) últimos ~2-3 anos selados, avaliados UMA vez no candidato congelado; (b) máximo de N levers/famílias; (c) kill-criterion numérico — se após a régua honesta (1-14) o DSR permanecer 0 com sample weights, abandonar a hipótese price-derived antes de gastar esforço alto em 18-20.
- risco: Baixo. Risco é disciplina: o lockbox só vale se realmente nunca for tocado em CPCV.


### Fase B — Dados + uniqueness (8–12)

**8. Ajuste de volume por split no painel (passo de dados próprio)** _(esforço baixo · dep [6])_
- hipótese: Volume não-ajustado por split cria saltos espúrios de 2-4× que viram falso sinal de liquidez/impacto. É pré-requisito ENTERRADO: alimenta o modelo de custos (|Q|/V), o Amihud e qualquer proxy de microestrutura/turnover. Tem que ser corrigido ANTES de custos e features, não dentro do passo de microestrutura.
- base SOTA: Qualidade de dados corporate-actions; validação contra COTAHIST (como já feito para preço no Lever 1).
- lift esperado: Nenhum lift; remove falso sinal de liquidez e torna custos/Amihud confiáveis.
- risco: Baixo.

**9. Universo point-in-time + deslistados (matar survivorship) — foundation de dados, não lever** _(esforço alto · dep [8])_
- hipótese: Universo = IBrX líquido de HOJE enviesa a base-rate de TODO label e retorno para cima (sobreviventes), contaminando toda a backtest. É correção de INTEGRIDADE de dados, não edge — pertence à fase da régua, ANTES de custos/sizing/objetivo. Colocá-la depois forçaria re-litigar todas as decisões econômicas sobre o chão trocado (o próprio efeito admite 'provável QUEDA do edge medido').
- base SOTA: Survivorship/PIT membership (López de Prado; prática padrão de asset pricing empírico); reconstrução de membership B3 + COTAHIST de deslistados.
- lift esperado: Provável QUEDA do edge medido — é a verdade. Habilita alt-data e expansão futura do universo.
- risco: Alto esforço (curadoria de membership + deslistados). Risco de bus-factor da ingestão manual brapi. Pagar agora evita re-tunar 13-20 sobre dados otimistas.

**10. Average uniqueness a partir do t1 (concorrência por barra)** _(esforço baixo · dep [1])_
- hipótese: Labels de 10d amostrados diariamente geram amostras redundantes (concurrency alta). uniqueness = 1/(concurrency+1) por observação é a base que falta para sample weights, sequential bootstrap E para desinflar o n do PSR/DSR. Maior ROI metodológico antes de qualquer modelo novo — revela que o N efetivo é ~10× menor que o nominal.
- base SOTA: AFML cap.4 — get_av_uniqueness_from_triple_barrier (mlfinlab).
- lift esperado: Nenhum lift isolado; habilita 11,12,14 e dimensiona corretamente toda a incerteza estatística downstream.
- risco: Baixo.

**11. Desinflar n do PSR/DSR (dedup + uniqueness), SR overlap-robusto e corrigir t-stat iid** _(esforço medio · dep [10])_
- hipótese: pooled=concat dos 15 splits duplica cada retorno ~5× (cada data em C(N-1,k-1) testes) e sobrepõe 10d → sqrt(n-1) infla maciçamente a confiança; skew/kurt viesados. Além disso meta_evaluate.py:105 trata os 15 splits CPCV como iid (t=mean/(std/sqrt(15))), o que é inválido para splits correlacionados. Deduplicar por (ticker,date), usar n = datas_únicas/HORIZON × avg_uniqueness e substituir o t-stat iid por inferência overlap-robusta torna o DSR robusto-conservador em vez de frágil-conservador.
- base SOTA: Bailey & López de Prado 2014 (PSR/DSR com N independente); AFML cap.4/14; correção de SR overlap-robusto.
- lift esperado: Nenhum lift; impede falso-positivo de DSR quando o sinal melhorar (16-20). Eleva o hurdle — é o objetivo. Critério de aceite confiável e numérico.
- risco: Médio. Pode elevar o hurdle e frustrar 'vitórias' aparentes — desejável.

**12. Log de trials de pesquisa + reconstruir os paths contíguos do CPCV** _(esforço medio · dep [11])_
- hipótese: Dois defeitos acoplados na semântica do DSR: (a) sr_star usa var dos 15 splits como variância de trials e n_trials=15 — mas paths CPCV são reamostragens de UMA config, não multiple-testing; logar os Sharpes de TODOS os configs/hiperparâmetros testados (L0-L4 + descartados) dá o sr_trials/n_trials corretos. (b) O código nunca reconstrói os C(N,k)·k/N=5 paths contíguos — sem eles não há distribuição real de backtest. Juntos: separam dispersão-do-estimador (paths) de deflação-por-trials (log), que hoje estão confundidas. O registro no log deve ser GATE automatizado, não disciplina manual, e cada família de features/hiperparâmetro entra nele.
- base SOTA: AFML cap.12 (combinatorial paths) + cap.14 (deflated Sharpe deflaciona por esforço de pesquisa real).
- lift esperado: Nenhum lift; corrige a semântica do DSR para detectar skill se houver e penalizar data-mining real.
- risco: Baixo-Médio. Requer instrumentar o log como gate; a deflação colapsa se algum trial escapar do registro.


### Fase C — Economia real (13–16)

**13. Modelo de custos de transação (lei da raiz quadrada) no backtest** _(esforço baixo · dep [1, 8])_
- hipótese: Sharpe long-short bruto é fantasia otimista por construção. Custo é PRÉ-CONDIÇÃO de qualquer claim de DSR — nenhum passo de modelo (14,16,17,20) pode reportar DSR como critério de aceite antes dele. Embutir G≈σ·(|Q|/V)^expoente (lei sqrt de impacto, Tóth) + turnover por trade transforma o número em P&L acreditável e define capacidade. Em mid-caps B3 de ADV baixo o impacto é mais convexo que em mercados desenvolvidos — tratar como PISO conservador, estressando expoente/σ nos nomes ilíquidos onde o long-short tende a alocar.
- base SOTA: Lei sqrt de market impact (Tóth et al. 2011) como base; Almgren-Chriss 2000 apenas como contexto de execução ótima, não como modelo de custo.
- lift esperado: Quase certamente NEGATIVO no Sharpe bruto — é o filtro que separa sinal tradeable de ruído de ranking. Reportar P&L líquido como único número econômico válido daqui em diante.
- risco: Médio. Pode matar o que sobrou do Sharpe; necessário saber antes de qualquer modelagem econômica.

**14. Sample weights (uniqueness × return-attribution) + sequential bootstrap no LightGBM** _(esforço medio · dep [10, 11])_
- hipótese: Sem sample_weight o modelo super-pondera regimes lotados (effective N << N), overfittando períodos super-amostrados; e o bootstrap padrão sorteia amostras redundantes, correlacionando as árvores. Ponderar por uniqueness×|retorno atribuído| (+ time-decay opcional) regulariza o fit, e o sequential bootstrap prioriza amostras de baixa sobreposição, aumentando a diversidade real do bagging. Ataca a causa-raiz do overfit a regimes. Só agora é avaliável: a régua honesta (11,12) e custos (13) já existem.
- base SOTA: AFML cap.4 — sample_weight por uniqueness e return attribution, time-decay, seq_bootstrap.
- lift esperado: AUC pode subir levemente ou cair (menos overfit); ganho real é estabilidade/honestidade. Resultado modal sob a régua endurecida é DSR≈0 — tratar DSR>0 como hipótese de baixa probabilidade, não alvo esperado.
- risco: Médio. Pode revelar que parte do AUC 0.523 era overfit a regimes. Custo computacional do seq_bootstrap ok para 28 tickers.

**15. Re-baseline obrigatório sobre o universo honesto (ponte pós-dados)** _(esforço baixo · dep [9, 13, 14])_
- hipótese: Quando survivorship (9), volume-split (8) e custos (13) corrigem o chão, as métricas de qualquer trabalho anterior ficam inválidas (base-rate e retornos mudam). Sem um passo explícito de re-execução, acumulam-se decisões sobre dados enviesados que nunca são reavaliadas. Re-rodar CPCV + custos + sample weights sobre o painel PIT re-confirma ou derruba os ganhos e congela o baseline honesto numérico que será o ponto-de-comparação dos levers 16-20.
- base SOTA: Princípio de re-baseline pós-correção de integridade de dados; pré-registro do hurdle (passo 7) aplicado ao baseline.
- lift esperado: Nenhum lift; estabelece o número honesto contra o qual todo lift futuro é medido. Provável que o edge caia frente ao baseline enviesado anterior.
- risco: Baixo. Desconforto de ver o edge encolher é o ponto.

**16. Bet sizing (sigmoide proba→tamanho; conformal como alternativa)** _(esforço baixo · dep [12, 13, 15])_
- hipótese: long>mediana/short<mediana joga fora a informação de ranking que o AUC capta. Converter proba em tamanho via z=(p-0.5)/sqrt(p(1-p)), size=2·N(z)-1 (conservador perto de 0.5, agressivo nas convicções, Kelly como teto) é onde o ranking pode vazar para P&L. Como a calibração da proba é duvidosa em N pequeno, avaliar sizing por intervalo conforme (conformal prediction) como alternativa mais defensável. Avaliar SEMPRE líquido de custos (13) e sob o n honesto (11).
- base SOTA: AFML cap.10 (bet sizing, Kelly como teto); conformal prediction 2024+ para sizing por incerteza calibrada em N pequeno.
- lift esperado: Potencial de traduzir ranking marginal em P&L sem custo de complexidade alto, MAS resultado modal é DSR=0 sob a régua honesta. Pode amplificar perdas se o ranking for ruído — por isso vem depois de custos, uniqueness e re-baseline.
- risco: Médio.


### Fase D — Modelo + features de fronteira (17–20)

**17. Learning-to-rank (LGBMRanker/lambdarank) com Rank IC pré-registrado** _(esforço medio · dep [12, 13, 15])_
- hipótese: O problema É cross-section ranking ('qual dos 28 trades dá mais certo HOJE'), modelado como classificação pointwise; AUC é cego à ordenação tradeable. lambdarank agrupado por data otimiza a ORDEM; avaliar por Rank IC, IC, Precision@N e spread de decis. CAVEAT estrutural: cada query tem só ~28 itens — sinal listwise raso e Rank IC diário de altíssima variância; o ganho de LTR vem de queries com centenas-milhares de itens, NÃO transfere para 28 nomes. Para evitar metric-shopping, Rank IC é pré-registrado como métrica primária de ranking ANTES de rodar (justificável a priori) e AUC+Rank IC são reportados lado a lado em TODOS os levers, não só na transição.
- base SOTA: LambdaMART (Burges 2010) / LGBMRanker nativo. NOTA: claims de '~3× Sharpe vs momentum' e 'Sharpe ~1.95 emerging markets' vêm de universos 10²-10³ ativos, sem custos, com survivorship próprio — tratados como folclore não-reproduzido, não como base.
- lift esperado: Teto BAIXO por construção (28 itens/query). Baseline interno honesto: melhor lift de feature já visto foi +0.0086 AUC (fracdiff) com DSR ainda 0. Exigir que a melhora sobreviva à deflação de trials (12) e seja líquida de custos; validar Rank IC com bootstrap por bloco.
- risco: Médio-Alto. Risco de eleger lambdarank vencedor no ruído de listas minúsculas; o arbitro econômico continua sendo DSR líquido, não o Rank IC.

**18. Eixos ortogonais de features: microestrutura (Amihud núcleo) + structural break (GSADF/CUSUM vol-robusto) + alt-data BTB** _(esforço alto · dep [9, 12, 13, 15])_
- hipótese: A família price-derived (TA+fracdiff) está exausta (platô ~0.52-0.523); edge novo precisa de eixos ORTOGONAIS, consolidados num único bloco para não explodir trials. (a) Microestrutura diária: Amihud (|ret|/turnover) é o único item genuinamente diário e robusto — núcleo viável; OFI/order-imbalance verdadeiro exige LOB/fluxo assinado (Cont-Kukanov-Stoikov), indisponível sem tick → rebaixado a proxy especulativo via tick-rule, não eixo principal. (b) Structural break: GSADF (supremo ADF expansivo) e CUSUM Chu-Stinchcombe-White capturam transições de regime ortogonais a momentum/vol — mesma família que rendeu o maior salto (fracdiff); CRÍTICO usar variante vol-robusta (séries B3 têm vol time-varying, testes clássicos assumem vol constante). (c) Alt-data BTB: open-interest diário de aluguel de ações é o candidato alt-data mais limpo e viável (short demand doméstico real). Cada sub-família entra no log de trials (12) como gate automatizado.
- base SOTA: Amihud 2002; Cont-Kukanov-Stoikov 2014 (OFI exige LOB); AFML cap.17 (SADF/GSADF) + CUSUM CSW + variantes vol-robustas (J. Financial Econometrics 2023); securities lending como proxy de short demand.
- lift esperado: Primeiros eixos genuinamente não-momentum, MAS expectativa honesta: 'pode ou não mover o DSR sob a régua', sem múltiplo de Sharpe externo. Opções de ação B3 descartadas do escopo (liquidez confinada a ~5 nomes → seleção/NaN-leak).
- risco: Médio-Alto. Sem tick não há OFI real; sem normalização de vol o structural break gera falso sinal; ingestão PIT de BTB é o elo menos escalável (risco de leak se não historizado). p/N_efetivo cresce — gated pelo kill-criterion (7).

**19. Clustered MDA/ONC como gate de feature-selection sobre o superset completo** _(esforço baixo · dep [18])_
- hipótese: ranks cross-section + fd_close/fd_vol21 + TA + microestrutura + structural break + BTB têm substitution effect forte; MDA padrão dilui importância entre features correlacionadas. Rodar clustered MDA (ONC) sob purged CV sobre o SUPERSET completo — não só price-derived — revela se cada eixo é robusto ou artefato de correlação e VETA colinearidade ANTES de alimentar o DL. Sem isso o conjunto final que entra no passo 20 nunca é vetado coletivamente.
- base SOTA: AFML cap.8 — clustered MDA/MDI, ONC; opcional denoising RMT / Ledoit-Wolf na camada de covariância do portfólio long-short.
- lift esperado: Nenhum lift direto; diagnóstico/gate barato que evita desperdiçar GPU em features redundantes e reduz a dimensionalidade efetiva antes do regime de overfit do DL.
- risco: Baixo.

**20. Modelagem small-N na GPU: TabPFN v2 primário; PatchTST encoder-as-feature condicional** _(esforço alto · dep [7, 12, 15, 17, 19])_
- hipótese: O constraint real é AMOSTRA (28 tickers × ~16y ≈ ~112k barras diárias), NÃO compute (VRAM 8GB sobra). TabPFN v2 é o SOTA tabular DESENHADO para N pequeno, roda em GPU modesta e é o uso de GPU com EV real para classificação/ranking cross-section de 28 nomes — ocupa o slot DL ANTES do transformer de séries longas. PatchTST entra apenas como experimento CONDICIONAL (se o kill-criterion do passo 7 autorizar): channel-independent + masked pretraining como gerador de embeddings/forecast-surprise alimentando o LightGBM/LTR, NUNCA como classificador standalone. LEAK CRÍTICO a fechar: o pré-treino masked não pode ver a série inteira (inclui janelas de teste de cada fold) — restringir o corpus por fold até o início purgado da janela de teste, OU pré-treinar só no complemento do lockbox. Scalers/embeddings por ticker ajustados SÓ dentro do fold (testado pela suíte do passo 6). Toda busca de hiperparâmetro entra no log de trials (12).
- base SOTA: TabPFN v2 (2024/2025) — SOTA tabular small-N, regime exato deste projeto. PatchTST 2023 (channel-independence + masked pretraining) como secundário. NOTA: 'Re(Visiting) TSFMs in Finance' 2025 espera AUC~0.51 e reprodução de momentum — contradiz tese de lift fácil; 'Kronos 2025' tratado como citação não-verificada, removido da base.
- lift esperado: Incerto e provavelmente baixo. EV positivo só porque os eixos saturados foram abertos antes e porque TabPFN substitui o transformer mal-dimensionado. Em 28 tickers ainda é regime de overfit. Aceite SOMENTE com DSR>0 líquido de custos, trials deflacionados E confirmação única no lockbox selado (7).
- risco: Alto. Overfit a painel pequeno, leak via pré-treino na série inteira / scaler-embedding global, auto-engano do DSR se trials não registrados. DL é o último passo, não o atalho — só se o kill-criterion não tiver abortado a hipótese price-derived antes.
