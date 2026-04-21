using System;

namespace perceptron
{
    class Program
    {
        static void Main(string[] args)
        {
            // {x1, x2, y}
            int[,] datos = { {1,1,0}, {1,0,1}, {0,1,1}, {0,0,0} };

            Random aleatorio = new Random();

            // pesos: w0 para x1, w1 para x2, w2 para x1*x2, w3 para bias
            double[] pesos = {
                aleatorio.NextDouble() - aleatorio.NextDouble(),
                aleatorio.NextDouble() - aleatorio.NextDouble(),
                aleatorio.NextDouble() - aleatorio.NextDouble(),
                aleatorio.NextDouble() - aleatorio.NextDouble()
            };

            bool aprendizaje = true;
            int salidaInt;
            int epocas = 0;
            double tasaAprendizaje = 0.1;

            while (aprendizaje && epocas < 10000)
            {
                aprendizaje = false;

                for (int i = 0; i < 4; i++)
                {
                    int x1 = datos[i, 0];
                    int x2 = datos[i, 1];
                    int y = datos[i, 2];

                    int x3 = x1 * x2; // término no lineal

                    double salidaDoub = x1 * pesos[0] +
                                        x2 * pesos[1] +
                                        x3 * pesos[2] +
                                        pesos[3];

                    if (salidaDoub > 0)
                        salidaInt = 1;
                    else
                        salidaInt = 0;

                    int error = y - salidaInt;

                    if (error != 0)
                    {
                        pesos[0] = pesos[0] + tasaAprendizaje * error * x1;
                        pesos[1] = pesos[1] + tasaAprendizaje * error * x2;
                        pesos[2] = pesos[2] + tasaAprendizaje * error * x3;
                        pesos[3] = pesos[3] + tasaAprendizaje * error * 1; // bias

                        aprendizaje = true;
                    }
                }

                epocas++;
            }

            // Pruebas
            for (int i = 0; i < 4; i++)
            {
                int x1 = datos[i, 0];
                int x2 = datos[i, 1];
                int y = datos[i, 2];

                int x3 = x1 * x2;

                double salidaDoub = x1 * pesos[0] +
                                    x2 * pesos[1] +
                                    x3 * pesos[2] +
                                    pesos[3];

                if (salidaDoub > 0)
                    salidaInt = 1;
                else
                    salidaInt = 0;

                Console.WriteLine("Entradas: " + x1 + " XOR " + x2 + " = " + y + " | Perceptron " + salidaInt);
            }

            Console.WriteLine("Épocas: " + epocas);
            Console.WriteLine("Pesos útiles: w0=" + pesos[0] + 
                            " w1=" + pesos[1] + 
                            " w2=" + pesos[2] + 
                            " bias=" + pesos[3]);
        }
    }
}