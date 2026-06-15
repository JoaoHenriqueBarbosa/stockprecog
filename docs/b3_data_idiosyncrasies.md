# Idiossincrasias dos dados da B3 — mapa completo

Documento de referência sobre como obter dados históricos **completos e ajustados**
de ações da B3 a partir de fontes 100% públicas e gratuitas. Resultado de
investigação empírica (não suposição).

## Veredito central

**Os dados ESTÃO disponíveis publicamente** — preços e proventos. A dificuldade
nunca foi disponibilidade; foi a **descontinuidade de identidade dos ativos** ao
longo do tempo (mudança de ticker/nome em reorganizações societárias).

## 1. Preços — COTAHIST (autoritativo, completo, grátis)

- ZIPs anuais: `https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{ANO}.ZIP`
- Layout posicional, registro "01", filtrar `CODBDI=02` (lote padrão) + `TPMERC=010` (à vista).
- Preços vêm ×100 (2 decimais implícitas).
- **É PREÇO BRUTO** — não ajustado por proventos/splits. Campo FATCOT ≈ sempre 1.
  Confirmado: nem o pacote maduro `rb3` (rOpenSci) ajusta automaticamente.

## 2. Proventos — API interna de empresas listadas (grátis, frágil)

- Endpoint: `sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall/GetListedCashDividends/{base64}`
  com base64 de `{language, pageNumber, pageSize, tradingName}`. pageSize máx 120, paginar. `verify=False` + UA de browser.
- Campos: `corporateAction` (DIVIDENDO / JRS CAP PROPRIO=JCP), `lastDatePriorEx`
  (último dia COM direito; ex-date = +1 dia), `valueCash`, `closingPricePriorExDate`, `typeStock` (ON/PN).
- **Fator de ajuste por evento** = `(close_prior − valueCash) / close_prior`,
  aplicado cumulativamente e retroativamente a todas as datas anteriores ao ex-date.
- Validação: ABEV3 reconstruído bate com yfinance dentro de ~2% (resíduo = JCP + convenções).

## 3. A GRANDE armadilha: descontinuidade de ticker/nome

A API de proventos indexa por **nome de pregão da época**, casado por **prefixo** —
não por ticker. Empresa que se reorganizou tem o histórico **partido entre nomes**.

- Caso canônico: **AmBev → Ambev S/A em out/2013, AMBV3/AMBV4 → ABEV3.**
  - `tradingName="AMBEV"` → proventos 2000–2013 (134 regs, preços ~R$84 pré-ajuste).
  - `tradingName="AMBEV S.A."` → 2013–2025 (38 regs).
  - Série completa = emendar os dois. COTAHIST também só tem ABEV3 a partir de 2013-11-11.
- `"AMBEV S/A"` (com barra) → 0. Tem que ser `"AMBEV S.A."` ou `"AMBEV SA"`. Busca por prefixo é instável.
- `codeCVM` e `issuingCompany` **não funcionam** no endpoint de dividendos (retornam 0).
- Filtrar `typeStock=ON` para a ação ordinária (ABEV3).

## 4. Cobertura da resolução automática de nome (teste em 15 tickers)

Resolver o nome via `GetInitialCompanies` (casando `issuingCompany==raiz`) cobre **~73%**:

| Funciona (11/15) | Falha (4/15) — precisa mapa manual |
|---|---|
| PETR4, VALE3, ITUB4 (948 regs), BBDC4, BBAS3, WEGE3, RENT3, SUZB3, CPLE6, RADL3, EQTL3 | BPAC11 (nome="BTGP BANCO" não casa dividendos), ELET3 (busca→None), EMBR3 (trunca p/ "EMBRAST"), PRIO3 (era PetroRio, só 1 reg sob nome novo) |

**Conclusão:** a resolução automática não é suficiente sozinha. Precisa de um
**dicionário manual de exceções** (ticker → nome(s) de dividendo, incluindo nomes
históricos pré-reorganização) para os ~27% que escapam.

## 5. Becos sem saída (não repetir)

- **API oficial "Área Logada do Investidor"** (developers.b3.com.br): dados por
  INVESTIDOR (posição/movimentação de um CPF consentido), exige consentimento +
  contrato PJ + self-assessment. **Não é fonte de histórico de mercado.**
- COTAHIST do ano corrente não marca proventos inline.
- yfinance `.SA`: tem preço ajustado mas rate-limita forte e descarta ~6 tickers
  líquidos como "possibly delisted" (falso). Serve como **referência de validação**, não fonte primária.

## 6. Survivorship bias

Composição point-in-time de índices **é pública** (templates `rb3`:
`b3-indexes-historical-data`, `b3-indexes-theoretical-portfolio`). Usar em vez da
composição atual do IBrX.
