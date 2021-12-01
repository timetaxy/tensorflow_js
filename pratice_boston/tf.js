`use strict`;
const tf = require('@tensorflow/tfjs');
const data = require('./data1');

//Hi there 👋. Looks like you are running TensorFlow.js in Node.js. To speed things up dramatically, install our node backend, which binds to TensorFlow C++, by running npm i @tensorflow/tfjs-node, or npm i @tensorflow/tfjs-node-gpu if you have CUDA. Then call require('@tensorflow/tfjs-node'); (-gpu suffix for CUDA) at the start of your program. Visit https://github.com/tensorflow/tfjs-node for more details.

// console.log(data.factor)
// console.log(data.result)

const dataFactor = data.factor;
const dataResult = data.Result;

const x = tf.input({ shape: [13] });
// const h1 = tf.layers.dense({units:13, activation:'relu'}).apply(x)
const y = tf.layers.dense({ units: 1 }).apply(x);
const model = tf.model({ inputs: x, outputs: y });
const compileParam = { optimizer: tf.train.adam(), loss: tf.losses.meanSquaredError };
model.compile(compileParam);

const fitParam = {
    epochs: 3000,
    callbacks: {
        onEpochEnd:
            function (epoch, logs) {
                console.log(`epoch:${epoch}, logs:${logs}, rmse:${Math.sqrt(logs.loss)}`);
            }
    }
};
model.fit(dataFactor[1], dataResult[1], fitParam).then(function (result) {
    let predition = mode.predict(dataFactor[1]);
    predition.print();
});

