/*
Copyright 2017 Tommy Petersen.

This file is part of "Learner".

"Learner" is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

"Learner" is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with "Learner".  If not, see <http://www.gnu.org/licenses/>. 
  */
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ANN;
using Dataset;

namespace Learner
{
    class MNISTLearner
    {
        List<int> layerSizes;
        ANN.ANN ann;
        Dataset.MNIST mnist;

        String trainingImagesFileName =
            "C:\\Users\\Tommy\\Documents\\" +
            "MachineLearning\\Dataset\\MNIST\\train-images.idx3-ubyte";
        String trainingLabelsFileName =
            "C:\\Users\\Tommy\\Documents\\" +
            "MachineLearning\\Dataset\\MNIST\\train-labels.idx1-ubyte";
        String testImagesFileName =
            "C:\\Users\\Tommy\\Documents\\" +
            "MachineLearning\\Dataset\\MNIST\\t10k-images.idx3-ubyte";
        String testLabelsFileName =
            "C:\\Users\\Tommy\\Documents\\" +
            "MachineLearning\\Dataset\\MNIST\\t10k-labels.idx1-ubyte";

        List<MNISTImage> trainingImages;
        List<byte> trainingLabels;
        List<MNISTImage> testImages;
        List<byte> testLabels;

        List<Matrix> trainingInputVectors;
        List<Matrix> trainingOutputVectors;
        List<Matrix> testInputVectors;
        List<Matrix> testOutputVectors;

        public List<MNISTImage> trainingImagesProp
        {
            get
            {
                return trainingImages;
            }
        }

        public List<Matrix> trainingInputVectorsProp
        {
            get
            {
                return trainingInputVectors;
            }
        }

        public List<byte> trainingLabelsProp
        {
            get
            {
                return trainingLabels;
            }
        }

        public List<Matrix> trainingOutputVectorsProp
        {
            get
            {
                return trainingOutputVectors;
            }
        }

        public void MNISTTrain(int miniBatchSize, double eta, int nrOfEpochs)
        {
            ann.trainANN(trainingInputVectors, trainingOutputVectors, miniBatchSize, eta, nrOfEpochs);
        }

        public byte MNISTClassify(MNISTImage image)
        {
            Matrix inputVector = convertImageToInputVector(image);
            //Feed forward:
            Tuple<List<Matrix>, List<Matrix>> zaLists = ann.feedForward(inputVector);
            List<Matrix> zList = zaLists.Item1;
            List<Matrix> aList = zaLists.Item2;

            Matrix outputVector = aList.Last<Matrix>();

            return convertOutputVectorToLabel(outputVector);
        }

        public double computePrecision()
        {
            int nrOfTestImages = testImages.Count();
            int nrOfCorrectGuesses = 0;
            byte labelGuess = 0;

            for (int i = 0; i < nrOfTestImages; i++)
            {
                labelGuess = MNISTClassify(testImages.ElementAt<MNISTImage>(i));
                if (labelGuess == testLabels.ElementAt<byte>(i))
                {
                    nrOfCorrectGuesses++;
                }
            }

            if (nrOfTestImages > 0)
            {
                return ((double) nrOfCorrectGuesses) / ((double) nrOfTestImages);
            }
            else
            {
                return 0D;
            }

        }

        public MNISTLearner(List<int> _layerSizes) {
            layerSizes = _layerSizes;
            ann = new ANN.ANN(layerSizes);
            mnist = new Dataset.MNIST();
            mnist.loadTrainingSetFiles(trainingImagesFileName, trainingLabelsFileName);
            mnist.loadTestSetFiles(testImagesFileName, testLabelsFileName);
            trainingImages = mnist.trainingImages;
            trainingLabels = mnist.trainingLabels;
            testImages = mnist.testImages;
            testLabels = mnist.testLabels;

            trainingInputVectors = convertImagesToInputVectors(trainingImages);
            trainingOutputVectors = convertLabelsToOutputVectors(trainingLabels);
            testInputVectors = convertImagesToInputVectors(testImages);
            testOutputVectors = convertLabelsToOutputVectors(testLabels);
        }

        public byte convertOutputVectorToLabel(Matrix outputVector)
        {
            byte label = 0;

            for (int m = 0; m < outputVector.M; m++)
            {
                if (outputVector[m, 0] > outputVector[label, 0])
                {
                    label = (byte) m;
                }
            }

            return label;
        }

        private Matrix convertImageToInputVector(MNISTImage image)
        {
            Matrix V = new Matrix(image.M * image.N, 1);

            for (int m = 0; m < image.M; m++)
            {
                for (int n = 0; n < image.N; n++)
                {
                    V[m * image.M + n, 0] = image[m, n] / 255D;
                }
            }

            return V;
        }

        private List<Matrix> convertImagesToInputVectors(List<MNISTImage> images)
        {
            List<Matrix> inputVectors = new List<Matrix>();

            foreach (MNISTImage image in images)
            {
                inputVectors.Add(convertImageToInputVector(image));
            }

            return inputVectors;
        }

        private List<Matrix> convertLabelsToOutputVectors(List<byte> labels)
        {
            List<Matrix> outputVectors = new List<Matrix>();

            Matrix V;
            foreach (byte label in labels)
            {
                V = new Matrix(10, 1);

                for (int j = 0; j < 10; j++)
                {
                    V[j, 0] = 0D;
                }

                V[label, 0] = 1D;

                outputVectors.Add(V);
            }

            return outputVectors;
        }
    }
}
