#[macro_use]
extern crate rulinalg;
extern crate rand;
use rand::{thread_rng, Rng};
use rulinalg::matrix::{BaseMatrix, Matrix};
use std::f32::consts::E;

struct Network {
    learning_rate: f32,
    weight_input_hidden: Matrix<f32>,
    bias_input: Matrix<f32>,
    weight_hidden_output: Matrix<f32>,
    bias_hidden: Matrix<f32>,
}

impl Network {
    pub fn new(num_input: usize, num_hidden: usize, num_output: usize, lr: f32) -> Network {
        let mut rng = thread_rng();
        let mut b_one = Vec::<f32>::new();
        let mut b_two = Vec::<f32>::new();
        for i in 0..num_hidden {
            let y: f32 = rng.gen_range(-1., 1.);
            b_one.push(y);
        }
        for i in 0..num_output {
            let y: f32 = rng.gen_range(-1., 1.);
            b_two.push(y);
        }
        Network {
            learning_rate: lr,
            weight_input_hidden: Matrix::<f32>::ones(num_hidden, num_input),
            bias_input: Matrix::new(1, num_hidden, b_one),
            weight_hidden_output: Matrix::<f32>::ones(num_output, num_hidden),
            bias_hidden: Matrix::new(1, num_output, b_two),
        }
    }

    pub fn guess(&self, input_vec: Matrix<f32>) -> Matrix<f32> {
        let mut hidden = &self.weight_input_hidden * input_vec.transpose();
        hidden += &self.bias_input;

        hidden = Matrix::new(
            hidden.rows(),
            hidden.cols(),
            hidden
                .iter()
                .map(|x| 1.0 / (1.0 + E.powf(-x)))
                .collect::<Vec<f32>>(),
        );

        let mut output = &self.weight_hidden_output * hidden;
        output += &self.bias_hidden;

        Matrix::new(
            output.rows(),
            output.cols(),
            output
                .iter()
                .map(|x| 1.0 / (1.0 + E.powf(-x)))
                .collect::<Vec<f32>>(),
        )
    }

    pub fn train(&mut self, input: Matrix<f32>, target: Matrix<f32>) {
        let mut hidden = &self.weight_input_hidden * input.transpose();
        hidden += &self.bias_input;

        let activated_hidden = Matrix::new(
            hidden.rows(),
            hidden.cols(),
            hidden
                .iter()
                .map(|x| 1.0 / (1.0 + E.powf(-x)))
                .collect::<Vec<f32>>(),
        );

        let mut output = &self.weight_hidden_output * &activated_hidden;
        output += &self.bias_hidden;

        let activated_output = Matrix::new(
            output.rows(),
            output.cols(),
            output
                .iter()
                .map(|x| 1.0 / (1.0 + E.powf(-x)))
                .collect::<Vec<f32>>(),
        );

        //TRAINING PART
        let error = target - &activated_output;
        let activated_minus_one = (&activated_output * -1.0) + 1.0;

        let gradient_hidden_output =
            &error.elemul(&activated_output.elemul(&activated_minus_one)) * self.learning_rate;

        self.weight_hidden_output += &gradient_hidden_output * activated_hidden.transpose();
        self.bias_hidden += gradient_hidden_output;

        //THIS MIGHT BE WHERE I MESS UP
        let hidden_error = self.weight_hidden_output.transpose() * error;

        let activated_hidden_minus_one = (&activated_hidden * -1.0) + 1.0;
        let gradient_input_hidden = &hidden_error
            .elemul(&activated_hidden.elemul(&activated_hidden_minus_one))
            * self.learning_rate;

        self.weight_input_hidden += &gradient_input_hidden * input;
        self.bias_input += gradient_input_hidden;

        //println!("{}", hidden_error);
        //println!("{}", activated_hidden);
        //println!("{}", activated_output);
        //println!("{}", self.weight_input_hidden);
        //println!("{}", self.weight_hidden_output);
        //println!("{}", self.bias_input);
        //println!("{}", self.bias_hidden);
    }
}

fn main() {
    let mut a = Network::new(2, 6, 1, 0.1);

    let data = vec![vec![1., 1.], vec![1., 0.], vec![0., 1.], vec![0., 0.]];
    let labels = vec![0.0, 1.0, 1.0, 0.0];

    let mut rng = thread_rng();
    for i in 0..20000 {
        let num = rng.gen_range(0, 4);
        a.train(
            Matrix::new(1, 2, data.get(num).unwrap().to_owned()),
            Matrix::new(1, 1, vec![labels.get(num).unwrap().to_owned()]),
        );
    }
    println!("True -- {}", a.guess(Matrix::new(1, 2, vec![1., 0.])));
    println!("False -- {}", a.guess(Matrix::new(1, 2, vec![1., 1.])));
    println!("False -- {}", a.guess(Matrix::new(1, 2, vec![0., 0.])));
    println!("True -- {}", a.guess(Matrix::new(1, 2, vec![0., 1.])));
}
