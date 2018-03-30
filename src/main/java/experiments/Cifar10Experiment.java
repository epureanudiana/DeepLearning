/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experiments;
import java.io.IOException;
import java.util.List;
import nl.tue.s2id90.dl.NN.Model;
import nl.tue.s2id90.dl.NN.activation.Activation;
import nl.tue.s2id90.dl.NN.activation.RELU;
import nl.tue.s2id90.dl.NN.activation.Sigmoid;
import nl.tue.s2id90.dl.NN.activation.Softmax;
import nl.tue.s2id90.dl.NN.initializer.Gaussian;
import nl.tue.s2id90.dl.NN.layer.Convolution2D;
import nl.tue.s2id90.dl.NN.layer.Flatten;
import nl.tue.s2id90.dl.NN.layer.FullyConnected;
import nl.tue.s2id90.dl.NN.layer.InputLayer;
import nl.tue.s2id90.dl.NN.layer.Layer;
import nl.tue.s2id90.dl.NN.layer.OutputSoftmax;
import nl.tue.s2id90.dl.NN.layer.PoolMax2D;
import nl.tue.s2id90.dl.NN.layer.SimpleOutput;
import nl.tue.s2id90.dl.NN.loss.CrossEntropy;
import nl.tue.s2id90.dl.NN.loss.MSE;
import nl.tue.s2id90.dl.NN.optimizer.Optimizer;
import nl.tue.s2id90.dl.NN.optimizer.SGD;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import nl.tue.s2id90.dl.NN.transform.DataTransform;
import nl.tue.s2id90.dl.NN.validate.Classification;
import nl.tue.s2id90.dl.NN.validate.Regression;
import nl.tue.s2id90.dl.experiment.Experiment;
import nl.tue.s2id90.dl.input.GenerateFunctionData;
import nl.tue.s2id90.dl.input.InputReader;
import nl.tue.s2id90.dl.input.MNISTReader;
import nl.tue.s2id90.dl.input.PrimitivesDataGenerator;
import nl.tue.s2id90.dl.javafx.FXGUI;
import nl.tue.s2id90.dl.javafx.ShowCase;
import experiments.GradientDescentWithMomentum;
import nl.tue.s2id90.dl.input.Cifar10Reader;

/**
 *
 * @author dianaepureanu
 */
public class Cifar10Experiment extends Experiment {
     // ( hyper ) parameters
    int batchSize = 50;
    int epochs = 5;
    float learningRate = 0.0001f;
    String [] labels= {"airplane" ,"automobile" ,"bird" ,"cat" ,"deer" ,
    "dog" ,"frog" ,"horse" ,"ship" ,"truck" };

    
    Cifar10Experiment(){ 
        super(true) ; 
    }

    public void go() throws IOException {
    
    
    Cifar10Reader reader =new Cifar10Reader( batchSize , 10) ;

    
    System.out.println("Reader info:\n" + reader.toString());
    TensorShape inputs = reader.getInputShape();
    int outputs = reader.getOutputShape().getNeuronCount();
    
    ////////////////// PREPROCESSING //////////////////////////// 
    List<TensorPair> myTrainingData = reader.getTrainingData();
    List<TensorPair> myValidationData = reader.getValidationData();
    
    DataTransform dt = new MeanSubtraction();
    dt.fit(myTrainingData);
    dt.transform(myTrainingData);
    dt.transform (myValidationData) ;
    ///////////////////////////////////////////////////
    //print one record
    reader.getValidationData(1).forEach(System.out:: println);
    Model m = createModel(inputs, outputs);
    m.initialize (new Gaussian());

    ShowCase showCase = new ShowCase( i  -> labels[i]); 
    FXGUI.getSingleton().addTab("show case", showCase.getNode()); 
    showCase.setItems(reader.getValidationData(100));
    
//    Optimizer sgd = SGD.builder().model(m).learningRate(learningRate)
//        .validator(new Classification()).build();
//    Optimizer sgd = SGD.builder().model(m).learningRate(learningRate)
//            .validator(new Classification())
//           .updateFunction(() -> new MyAdaDelta(() ->
//            new L2Decay(GradientDescentWithMomentum ::new, 0.0001f)))
//            .build();
    Optimizer sgd = SGD.builder().model(m).learningRate(learningRate)
            .validator(new Classification())
            //.updateFunction(() -> new L2Decay(GradientDescentWithMomentum ::new, 0.0001f))
            //.updateFunction(() -> new MyAdaDelta(GradientDescentWithMomentum ::new))
            .updateFunction(() -> new MyAdaDelta(() ->
                    new L2Decay(GradientDescentWithMomentum ::new, 0.0001f)))
            .build();
    
            //.updateFunction(GradientDescentWithMomentum ::new).build() ;
    trainModel(m, reader, sgd, epochs, 0);
    }
    
    
    Model createModel ( TensorShape inputs , int outputs ) {
        
    int kernelSize = 1;
    int noFilters = 1;
    Activation activation = new RELU();
   //input
    InputLayer iLayer = new InputLayer("In",inputs, true);
    Model model = new Model(iLayer);
    //2d
    Layer convolutional =new Convolution2D("Convolution", iLayer.getOutputShape(), kernelSize,
            noFilters, activation);    
    model.addLayer(convolutional);
    
    //pooling
    Layer pool = new PoolMax2D("Pool", convolutional.getOutputShape(), 1);
    model.addLayer(pool);
    //2d
//    Layer convolutional2 =new Convolution2D("Convolution", pool.getOutputShape(), kernelSize,
//            noFilters, activation);    
//    model.addLayer(convolutional2);
//    
//    //pool
//    Layer pool2 = new PoolMax2D("Pool", convolutional2.getOutputShape(), 1);
//    model.addLayer(pool2);
    // add flatten layer after input layer
    Layer flatter = new Flatten ("Flatten", pool.getOutputShape());
    model.addLayer(flatter);
    
    //fully
    Layer fully = new FullyConnected("fc1", flatter.getOutputShape(), 
           inputs.getNeuronCount(), new RELU());
    model.addLayer(fully);
    //fully2
//    Layer fully2 = new FullyConnected("fc1", fully.getOutputShape(), inputs.getNeuronCount(), new RELU());
//    model.addLayer(fully2);
    
    //output
    Layer output = new OutputSoftmax("Out", fully.getOutputShape(), outputs, new CrossEntropy());
    
    System.out.format("conv: %s and pool: %s  and flatter: %s and output: %s \n", convolutional.getOutputShape(), 
            pool.getOutputShape(), flatter.getOutputShape(), output.getOutputShape() );
    
    model.addLayer(output);

    return model ;
}
    
    public static void main(String[] args) throws IOException {
        new Cifar10Experiment().go();
   }
}
