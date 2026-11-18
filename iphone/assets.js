/* Pinned, free model files. They are downloaded once; inference stays on device. */
(function (root) {
  const base = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21';
  root.PREX_ASSETS = Object.freeze({
    version: 'iphone-observation-2.0.0',
    runtime: base + '/vision_bundle.mjs',
    wasmBase: base + '/wasm',
    hand: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
    bottle: 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite',
    files: [
      base + '/vision_bundle.mjs',
      ...['vision_wasm_internal.js', 'vision_wasm_internal.wasm',
        'vision_wasm_nosimd_internal.js', 'vision_wasm_nosimd_internal.wasm'].map(n => base + '/wasm/' + n),
      'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite'
    ]
  });
})(globalThis);
