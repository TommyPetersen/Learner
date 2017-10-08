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
    class testMNISTLearner
    {
        static void Main(string[] args)
        {
            List<int> layerSizes = new List<int>();
            layerSizes.Add(28 * 28);
            layerSizes.Add(30);
            layerSizes.Add(10);

            MNISTLearner learner = new MNISTLearner(layerSizes);

            List<byte> trainingLabels = learner.trainingLabelsProp;
            List<Matrix> trainingOutputVectors = learner.trainingOutputVectorsProp;

            List<MNISTImage> trainingImages = learner.trainingImagesProp;
            List<Matrix> trainingInputVectors = learner.trainingInputVectorsProp;

            Console.WriteLine("trainingOutputVectors.count: " + trainingOutputVectors.Count());
            Console.WriteLine("trainingInputVectors.count: " + trainingInputVectors.Count());

            Random random = new Random();

            int i;

            Console.WriteLine("Enter non-empty line to quit!");
            while (false && Console.ReadLine().Equals(""))
            {
                i = random.Next(60000);
                Console.WriteLine("trainingLabels[" + i + "]:");
                Console.WriteLine(trainingLabels[i]);
                Console.WriteLine("trainingOutputVectors[" + i + "]:");
                Console.WriteLine(trainingOutputVectors[i].ToString());

                Console.WriteLine("Enter to test Image/Input!");
                Console.ReadLine();

                Console.WriteLine("trainingImages[" + i + "]:");
                Console.WriteLine(trainingImages[i]);
                Console.WriteLine("trainingInputVectors[" + i + "]:");
                Console.WriteLine((~trainingInputVectors[i]).ToString());
                Console.WriteLine("M: " + trainingInputVectors[i].M);
                Console.WriteLine("N: " + trainingInputVectors[i].N);

                Console.WriteLine("Enter non-empty line to proceed!");
            }

            Matrix OV = new Matrix(10, 1);
            OV[0, 0] = 0.123;
            OV[1, 0] = 0.432;
            OV[2, 0] = 0.998;
            OV[3, 0] = 0.768;
            OV[4, 0] = 0.543;
            OV[5, 0] = 0.432;
            OV[6, 0] = 0.359;
            OV[7, 0] = 0.819;
            OV[8, 0] = 1.039;
            OV[9, 0] = 2.591;
            
            byte convertedLabel = learner.convertOutputVectorToLabel(OV);

            Console.WriteLine("Converted label: " + convertedLabel);

            int miniBatchSize = 10;
            double eta = 3D;
            int nrOfEpochs = 1;

            Console.WriteLine("Training network...");
            learner.MNISTTrain(miniBatchSize, eta, nrOfEpochs);
            Console.WriteLine("...done");

            Console.WriteLine("Precision = " + learner.computePrecision());

            Console.WriteLine("testLearner: DONE!");
        }
    }
}
