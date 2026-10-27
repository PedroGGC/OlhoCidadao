/** Inferência idêntica à versão NumPy, incluindo transformação e calibração. */
export function predictMlpBrowser(expense, modelParams) {
  const { mean, std, W1, b1, W2, b2, featureTransform, calibration, decisionThreshold } = modelParams;
  let features;
  if (featureTransform?.version === 2) {
    const value = Math.log1p(Math.max(0, expense.valorLiquido));
    const code = Number(expense.codTipoDespesa);
    const [center, scale] = featureTransform.context[String(code)] || featureTransform.fallback;
    const known = featureTransform.categories.includes(code);
    features = [value, (value - center) / scale,
      ...featureTransform.categories.map((category) => Number(category === code)), Number(!known)];
  } else {
    features = expense; // compatibilidade com o payload anterior
  }
  if (features.length !== W1.length) throw new Error('Entradas incompatíveis com a rede treinada');
  const normalized = features.map((value, i) => (value - mean[i]) / (std[i] + 1e-8));
  const a1 = b1.map((bias, j) => {
    let z = bias;
    for (let i = 0; i < normalized.length; i++) z += normalized[i] * W1[i][j];
    return modelParams.activation === 'leaky_relu' ? (z > 0 ? z : .01 * z) : Math.tanh(z);
  });
  const logits = b2.map((bias, k) => {
    let value = bias;
    for (let j = 0; j < a1.length; j++) value += a1[j] * W2[j][k];
    return value;
  });
  const logit = (calibration?.slope ?? 1) * (logits[1] - logits[0]) + (calibration?.intercept ?? 0);
  const probability = logit >= 0 ? 1 / (1 + Math.exp(-logit)) : Math.exp(logit) / (1 + Math.exp(logit));
  return { classePredita: Number(probability >= (decisionThreshold ?? .5)), probabilidadeAtipico: probability, a1 };
}
