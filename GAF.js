// 
function data_processing(data_array, label_array, forward_lag, window_size, rolling_size) {

    var n_batch = Math.floor((data_array.length - window_size) / rolling_size);
    console.log(n_batch);
    var batch = new Array;
    var batch_label = new Array;
    var batch_index = new Array;
    var rolling_start = 0;
    for (i_batch = 0; i_batch < n_batch; i_batch++) {
        var this_batch = new Array;
        var this_label = label_array[rolling_start + window_size - 1 + forward_lag];
        for (i_window = rolling_start; i_window < rolling_start + window_size; i_window++) {
            this_batch.push(data_array[i_window]);
        }
        batch.push(this_batch);
        batch_label.push(this_label);
        batch_index.push(rolling_start + window_size - 1);
        rolling_start = rolling_start + rolling_size;
    }
    return [batch, batch_label, batch_index];
}

function gramian_cos_matrix(data_array) {

    var max_data = -9999;
    var min_data = 9999;
    for (i_data = 0; i_data < data_array.length; i_data++) {
        if (data_array[i_data] > max_data) {
            max_data = data_array[i_data];
        }
        if (data_array[i_data] < min_data) {
            min_data = data_array[i_data];
        }
    }

    var normalized_data = new Array;
    for (i_data = 0; i_data < data_array.length; i_data++) {
        normalized_data.push((data_array[i_data] - min_data) / (max_data - min_data));
    }

    var theta_array = new Array;
    for (i_data = 0; i_data < data_array.length; i_data++) {
        theta_array.push(Math.acos(normalized_data[i_data]));
    }

    var gramian_data_array = new Array;
    for (i_data = 0; i_data < data_array.length; i_data++) {
        var row_gramian_data = new Array;
        for (j_data = 0; j_data < data_array.length; j_data++) {
            row_gramian_data.push(Math.cos(Math.abs(Math.abs(theta_array[i_data]) + theta_array[j_data])));
        }
        gramian_data_array.push(row_gramian_data);
    }

    return gramian_data_array;
}

function gramian_cos_batch(batch_data) {

    var n_batch = batch_data.length;
    var batch_gramian_matrix = new Array;
    for (i_batch = 0; i_batch < n_batch; i_batch++) {
        batch_gramian_matrix.push(gramian_cos_matrix(batch_data[i_batch]));
    }
    return batch_gramian_matrix;
}


function gramian_cos_inv(weight_array) {

    var weight_size = Math.sqrt(weight_array.length);
    var feature_array = new Array;

    for (i = 0; i < weight_size; i++) {
        feature_array.push(0.5 * Math.acos(weight_array[i * weight_size + i]));
    }

    return feature_array;
}


function fitness_value(batch_data, feature_array) {

    var data_depth = batch_data.length;
    var matrix_size = batch_data[0].length;
    console.log('data_depth:' + data_depth);
    console.log('matrix_size' + matrix_size);

    var feature_size = Math.sqrt(feature_array.length);

    var fitness_value;
    var fitness_batch = new Array;

    for (i_batch = 0; i_batch < data_depth; i_batch++) {
        this_batch_data = batch_data[i_batch];
        fitness_value = 0;
        batch_start = matrix_size - feature_size;
        for (i_feature = 0; i_feature < feature_size; i_feature++) {
            for (j_feature = 0; j_feature < feature_size; j_feature++) {
                fitness_value += feature_array[i_feature * feature_size + j_feature] * this_batch_data[batch_start + i_feature][batch_start + j_feature];
            }
        }

        fitness_batch.push(fitness_value);

    }

    console.log(fitness_batch);

    return fitness_batch;
}


function lenet_start(net, batch_data, fitness_batch, label_data, batch_index) {

    var data_depth = batch_data.length;
    var matrix_size = batch_data[0].length;

    console.log('data_depth:', data_depth);
    console.log('matrix_size:', matrix_size);


    // this gets executed on startup
    var layer_defs = [];
    // input layer of size 1x1x2 (all volumes are 3D)
    //layer_defs.push({ type: 'input', out_sx: matrix_size, out_sy: matrix_size, out_depth: 1 });

    // matrix_size x matrix_size x 1
    layer_defs.push({ type: 'input', out_sx: matrix_size, out_sy: matrix_size, out_depth: 1 });
    layer_defs.push({ type: 'conv', sx: 10, filters: 4, stride: 1, ctivation: 'relu' });
    layer_defs.push({ type: 'pool', sx: 2, stride: 1 });
    //layer_defs.push({ type: 'conv', sx: 5, filters: 4, stride: 1, activation: 'relu' });
    //layer_defs.push({ type: 'pool', sx: 2, stride: 3 });
    layer_defs.push({ type: 'fc', num_neurons: 10, activation: 'sigmoid' });
    layer_defs.push({ type: 'softmax', num_classes: class_number });

    // create a net out of it
    var net = new convnetjs.Net();
    net.makeLayers(layer_defs);
    var x = new convnetjs.Vol(matrix_size, matrix_size, 1);

    var x = new convnetjs.Vol(matrix_size, matrix_size, 1);

    for (i_data = 0; i_data < matrix_size; i_data++) {
        for (j_data = 0; j_data < matrix_size; j_data++) {
            x.w[i_data * matrix_size + j_data] = batch_data[0][i_data][j_data];
        }
    }

    var trainer = new convnetjs.SGDTrainer(net, { learning_rate: 0.01, momentum: 0.01, batch_size: 1, l2_decay: 0.01 });

    trainer.train(x, label_data[0]);

    var pred_prob = net.forward(x);
    var pred = 0;
    for (i = 0; i < class_number; i++) {
        pred += pred_prob.w[i] * i;
    }

    console.log('pred :' + pred + ', err:' + (Math.abs(pred - label_data[0])));


    //setInterval(lenet_periodic(batch_data, label_data,net,trainer), 1000);
    lenet_train = setInterval(function() { lenet_periodic(net, trainer, batch_data, fitness_batch, label_data, batch_index); }, 5000);

}


function lenet_periodic(net, trainer, batch_data, fitness_batch, label_data, batch_index) {


    var pred_data = new Array;
    var batch_size = batch_data.length;
    var matrix_size = batch_data[0].length;
    var total_err = 0;
    var iter = Math.max(1, global_error.length - 1);
    var converge = false;
    if (iter > 10) {
        if (global_error[global_error.length - 1] < 0.001) {
            converge = true;
        }
    }

    if (converge == false) {
        for (i_batch = 0; i_batch < batch_size; i_batch++) {

            var x = new convnetjs.Vol(matrix_size, matrix_size, 1);
            for (i_data = 0; i_data < matrix_size; i_data++) {
                for (j_data = 0; j_data < matrix_size; j_data++) {
                    x.w[i_data * matrix_size + j_data] = batch_data[i_batch][i_data][j_data];
                }
            }

            trainer.train(x, label_data[i_batch]);

            var pred_prob = net.forward(x);
            var pred = 0;
            for (i = 0; i < class_number; i++) {
                pred += pred_prob.w[i] * i;
            }
            total_err += Math.pow(pred - label_data[i_batch], 2);
            //console.log('pred :' + pred + ', label:' + label_data[i_batch] + ', err:' + (Math.abs(pred - label_data[i_batch])));

            pred_data.push(pred);
            //console.log('total_err cum...'+total_err);
        }
        global_error.push(total_err / batch_size);
        console.log('iteration :' + global_error.length);
        console.log('error :' + global_error[global_error.length - 1]);
        var err_index = Array.from(Array(global_error.length).keys());


        var errConfig = {
            type: 'line',
            data: {
                labels: err_index,
                datasets: [{
                    label: 'error',
                    borderColor: 'rgba(100, 0, 0, 1)',
                    backgroundColor: 'rgba(100, 0, 0, 1)',
                    radius: 0, // radius is 0 for only this dataset
                    fill: false,
                    data: global_error
                }]
            }
        }

        //Get context with jQuery - using jQuery's .get() method.
        var errctx = $("#errPlot").get(0).getContext("2d");
        //This will get the first returned node in the jQuery collection.
        new Chart(errctx, errConfig);



        var predConfig = {
            type: 'line',
            animation: false, // Edit: correction typo: from 'animated' to 'animation'
            data: {
                labels: batch_index,
                datasets: [{
                        label: 'rolling pred',
                        borderColor: 'rgba(0, 0, 255, 1)',
                        backgroundColor: 'rgba(0, 0, 255, 1)',
                        radius: 0, // radius is 0 for only this dataset
                        fill: false,
                        data: pred_data
                    },
                    {
                        label: 'rolling label',
                        borderColor: 'rgba(120, 150, 0, 0.3)',
                        backgroundColor: 'rgba(120, 150, 000, 0.3)',
                        radius: 0, // radius is 0 for only this dataset
                        fill: false,
                        data: label_data
                    }
                ]
            }
        }

        //Get context with jQuery - using jQuery's .get() method.
        var predctx = $("#predPlot").get(0).getContext("2d");
        //This will get the first returned node in the jQuery collection.
        new Chart(predctx, predConfig);

    } else {

        clearInterval(lenet_train);

        console.log('done!');

    }


    sample_feature = gramian_cos_inv(net.layers[1].filters[0].w);

    var feature_index = Array.from(Array(sample_feature.length + 1).keys());
    feature_index = feature_index.slice(1, feature_index.length);

    var featureConfig = {
        type: 'line',
        data: {
            labels: feature_index,
            datasets: [{
                label: 'feature',
                borderColor: 'rgba(0, 150, 0, 1)',
                backgroundColor: 'rgba(0, 150, 000, 1)',
                radius: 0, // radius is 0 for only this dataset
                fill: false,
                data: sample_feature
            }]
        },
        options: {
            scales: {
                xAxes: [{
                    display: false
                }]
            }
        }
    }


    

    //Get context with jQuery - using jQuery's .get() method.
    var featureCtx = $("#feature1Plot").get(0).getContext("2d");
    //This will get the first returned node in the jQuery collection.
    new Chart(featureCtx, featureConfig);

    fitness_data = fitness_value(fitness_batch, net.layers[1].filters[0].w);

    var fitness_index = Array.from(Array(fitness_data.length + 1).keys());
    fitness_index = fitness_index.slice(1, fitness_index.length);

    var fitnessConfig = {
        type: 'line',
        data: {
            labels: fitness_index,
            datasets: [{
                label: 'fitness(before softmax)',
                borderColor: 'rgba(110, 150, 0, 1)',
                backgroundColor: 'rgba(110, 150, 000, 1)',
                radius: 0, // radius is 0 for only this dataset
                fill: false,
                data: fitness_data
            }]
        },
        options: {
            scales: {
                xAxes: [{
                    display: false
                }],
                yAxes: [{
                    display: true,
                    ticks: {
                        min: -2,
                        max: 2
                    }
                }]
            }
        }
    }

    //Get context with jQuery - using jQuery's .get() method.
    var fitnessCtx = $("#fitness1Plot").get(0).getContext("2d");
    //This will get the first returned node in the jQuery collection.
    new Chart(fitnessCtx, fitnessConfig);


     sample_feature = gramian_cos_inv(net.layers[1].filters[1].w);

    var feature_index = Array.from(Array(sample_feature.length + 1).keys());
    feature_index = feature_index.slice(1, feature_index.length);

    var featureConfig = {
        type: 'line',
        data: {
            labels: feature_index,
            datasets: [{
                label: 'feature',
                borderColor: 'rgba(0, 150, 0, 1)',
                backgroundColor: 'rgba(0, 150, 000, 1)',
                radius: 0, // radius is 0 for only this dataset
                fill: false,
                data: sample_feature
            }]
        },
        options: {
            scales: {
                xAxes: [{
                    display: false
                }]
            }
        }
    }


    

    //Get context with jQuery - using jQuery's .get() method.
    var featureCtx = $("#feature2Plot").get(0).getContext("2d");
    //This will get the first returned node in the jQuery collection.
    new Chart(featureCtx, featureConfig);

    fitness_data = fitness_value(fitness_batch, net.layers[1].filters[1].w);

    var fitness_index = Array.from(Array(fitness_data.length + 1).keys());
    fitness_index = fitness_index.slice(1, fitness_index.length);

    var fitnessConfig = {
        type: 'line',
        data: {
            labels: fitness_index,
            datasets: [{
                label: 'fitness(before softmax)',
                borderColor: 'rgba(110, 150, 0, 1)',
                backgroundColor: 'rgba(110, 150, 000, 1)',
                radius: 0, // radius is 0 for only this dataset
                fill: false,
                data: fitness_data
            }]
        },
        options: {
            scales: {
                xAxes: [{
                    display: false
                }],
                yAxes: [{
                    display: true,
                    ticks: {
                        min: -2,
                        max: 2
                    }
                }]
            }
        }
    }

    //Get context with jQuery - using jQuery's .get() method.
    var fitnessCtx = $("#fitness2Plot").get(0).getContext("2d");
    //This will get the first returned node in the jQuery collection.
    new Chart(fitnessCtx, fitnessConfig);


    ///////

     sample_feature = gramian_cos_inv(net.layers[1].filters[2].w);

    var feature_index = Array.from(Array(sample_feature.length + 1).keys());
    feature_index = feature_index.slice(1, feature_index.length);

    var featureConfig = {
        type: 'line',
        data: {
            labels: feature_index,
            datasets: [{
                label: 'feature',
                borderColor: 'rgba(0, 150, 0, 1)',
                backgroundColor: 'rgba(0, 150, 000, 1)',
                radius: 0, // radius is 0 for only this dataset
                fill: false,
                data: sample_feature
            }]
        },
        options: {
            scales: {
                xAxes: [{
                    display: false
                }]
            }
        }
    }
    

    //Get context with jQuery - using jQuery's .get() method.
    var featureCtx = $("#feature3Plot").get(0).getContext("2d");
    //This will get the first returned node in the jQuery collection.
    new Chart(featureCtx, featureConfig);

    fitness_data = fitness_value(fitness_batch, net.layers[1].filters[2].w);

    var fitness_index = Array.from(Array(fitness_data.length + 1).keys());
    fitness_index = fitness_index.slice(1, fitness_index.length);

    var fitnessConfig = {
        type: 'line',
        data: {
            labels: fitness_index,
            datasets: [{
                label: 'fitness(before softmax)',
                borderColor: 'rgba(110, 150, 0, 1)',
                backgroundColor: 'rgba(110, 150, 000, 1)',
                radius: 0, // radius is 0 for only this dataset
                fill: false,
                data: fitness_data
            }]
        },
        options: {
            scales: {
                xAxes: [{
                    display: false
                }],
                yAxes: [{
                    display: true,
                    ticks: {
                        min: -2,
                        max: 2
                    }
                }]
            }
        }
    }

    //Get context with jQuery - using jQuery's .get() method.
    var fitnessCtx = $("#fitness3Plot").get(0).getContext("2d");
    //This will get the first returned node in the jQuery collection.
    new Chart(fitnessCtx, fitnessConfig);


    /////////////

     sample_feature = gramian_cos_inv(net.layers[1].filters[3].w);

    var feature_index = Array.from(Array(sample_feature.length + 1).keys());
    feature_index = feature_index.slice(1, feature_index.length);

    var featureConfig = {
        type: 'line',
        data: {
            labels: feature_index,
            datasets: [{
                label: 'feature',
                borderColor: 'rgba(0, 150, 0, 1)',
                backgroundColor: 'rgba(0, 150, 000, 1)',
                radius: 0, // radius is 0 for only this dataset
                fill: false,
                data: sample_feature
            }]
        },
        options: {
            scales: {
                xAxes: [{
                    display: false
                }]
            }
        }
    }
    

    //Get context with jQuery - using jQuery's .get() method.
    var featureCtx = $("#feature4Plot").get(0).getContext("2d");
    //This will get the first returned node in the jQuery collection.
    new Chart(featureCtx, featureConfig);

    fitness_data = fitness_value(fitness_batch, net.layers[1].filters[3].w);

    var fitness_index = Array.from(Array(fitness_data.length + 1).keys());
    fitness_index = fitness_index.slice(1, fitness_index.length);

    var fitnessConfig = {
        type: 'line',
        data: {
            labels: fitness_index,
            datasets: [{
                label: 'fitness(before softmax)',
                borderColor: 'rgba(110, 150, 0, 1)',
                backgroundColor: 'rgba(110, 150, 000, 1)',
                radius: 0, // radius is 0 for only this dataset
                fill: false,
                data: fitness_data
            }]
        },
        options: {
            scales: {
                xAxes: [{
                    display: false
                }],
                yAxes: [{
                    display: true,
                    ticks: {
                        min: -2,
                        max: 2
                    }
                }]
            }
        }
    }

    //Get context with jQuery - using jQuery's .get() method.
    var fitnessCtx = $("#fitness4Plot").get(0).getContext("2d");
    //This will get the first returned node in the jQuery collection.
    new Chart(fitnessCtx, fitnessConfig);

    }

function softmax(arr) {
    return arr.map(function(value, index) {
        return Math.exp(value) / arr.map(function(y /*value*/ ) { return Math.exp(y) }).reduce(function(a, b) { return a + b })
    })
}