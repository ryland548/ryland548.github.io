const predictOutput = document.querySelector('#predict-text');
const predictButton = document.querySelector('#predictBtn');
predictOutput.innerText = 'NIL';

let session = null;

async function loadModel() {
  try {
    session = await ort.InferenceSession.create('./mnist_model.onnx', {
      externalData: [
        {
          path: './mnist_model.onnx.data',
          data: './mnist_model.onnx.data'
        },
        {
          path: 'mnist_model.onnx.data',
          data: './mnist_model.onnx.data'
        }
      ]
    });
    console.log('ONNX model loaded successfully with external weights!');
  } catch (e) {
    console.error('Failed to load ONNX model:', e);
  }
}
loadModel();

predictButton.addEventListener('click', async () => {
  if (!session) {
    alert('Model is still loading, please wait...');
    return;
  }
  const imageData = ctx.getImageData(0, 0, 28, 28);
  const data = imageData.data;
  const float32Data = new Float32Array(28 * 28);

  for (let i = 0; i < data.length; i += 4) {
    const alpha = data[i + 3];
    float32Data[i / 4] = alpha / 255.0;
  }
  const inputTensor = new ort.Tensor('float32', float32Data, [1, 1, 28, 28]);
  const feeds = { input: inputTensor };
  const results = await session.run(feeds);
  const outputData = results.output.data;
  let predictedDigit = 0;
  let maxScore = outputData[0];

  for (let i = 1; i < outputData.length; i++) {
    if (outputData[i] > maxScore) {
      maxScore = outputData[i];
      predictedDigit = i;
    }
  }
  predictOutput.innerText = predictedDigit;
});
