#![warn(non_snake_case)]
use itertools::iproduct;
use rand::rngs::StdRng;
use rand::Rng;
use std::f32::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};

//mod bak;

const KB: f64 = 1.3806e-23;
const NA: f64 = 6.022e23;

const NUM_TIME_STEPS: i32 = 10000;
const DT_STAR: f64 = 0.001;

const N: i32 = 16384;
const SIGMA: f64 = 3.405;
const EPSILON: f64 = 1.654e-21;
const EPS_STAR: f64 = EPSILON / KB;

const RHOSTAR: f64 = 0.6;
const RHO: f64 = RHOSTAR / (SIGMA * SIGMA * SIGMA);
const R_CUTOFF: f64 = SIGMA * 2.5;
const R_CUTOFF_SQUARED: f64 = R_CUTOFF * R_CUTOFF;
const T_STAR: f64 = 1.24;
const TARGET_TEMP: f64 = T_STAR * EPS_STAR;
const MASS: f64 = 39.9 * 10. / NA / KB;
const TARGET_CELL_LENGTH: f64 = R_CUTOFF;

fn main() {
    let sim_length = f64::cbrt(N as f64 / RHO);
    let cells_per_dimension = f64::floor(sim_length / TARGET_CELL_LENGTH);
    let cell_length = sim_length / cells_per_dimension;
    let cells_2d = cells_per_dimension * cells_per_dimension;
    let cells_3d = cells_per_dimension * cells_2d;
    println!("{} cells per dimension", cells_per_dimension);
    println!("{} atoms", N);
    println!("{} cells overall", cells_3d);
    println!("{} cell length", cell_length);
    let mut f = BufWriter::new(File::create("rusty.xyz").unwrap());
    let mut dbg_file = BufWriter::new(File::create("dbg.xyz").unwrap());

    let mut ke: Vec<f64> = Vec::new();
    let mut pe: Vec<f64> = Vec::new();
    let mut total_e: Vec<f64> = Vec::new();

    let mut accelerations: [[f64; 3]; N as usize] = [[0.0; 3]; N as usize];
    let mut old_accelerations: [[f64; 3]; N as usize] = [[0.0; 3]; N as usize];
    let mut positions = face_centered_cell();
    let mut velocities: [[f64; 3]; N as usize] = [[0.0; 3]; N as usize];
    let mut rng: StdRng = rand::SeedableRng::from_seed([3; 32]);

    for element in velocities.iter_mut().flatten() {
        *element = rng.gen_range(-1.0..1.0);
    }

    write_dbg(
        &positions,
        &velocities,
        &accelerations,
        &old_accelerations,
        &mut dbg_file,
        0,
    );
    thermostat(&mut velocities);

    let time_step = DT_STAR * f64::sqrt(MASS * SIGMA * SIGMA / EPS_STAR);
    let mut count = 0.05;
    for time in 0..NUM_TIME_STEPS {
        if time as f64 > count * NUM_TIME_STEPS as f64 {
            println!("{}%", (count * 100.) as i32);
            count += 0.05;
        }

        write_positions(&positions, &mut f, time);
        write_dbg(
            &positions,
            &velocities,
            &accelerations,
            &old_accelerations,
            &mut dbg_file,
            time,
        );

        for (idx, pos) in positions.iter_mut().enumerate() {
            for i in 0..3 {
                pos[i] += velocities[idx][i] * time_step
                    + 0.5 * accelerations[idx][i] * time_step * time_step;
                pos[i] += -sim_length * f64::floor(pos[i] / sim_length);
            }
        }
        old_accelerations = accelerations;

        accelerations = [[0.0; 3]; N as usize];
        let net_potential = calc_forces(&positions, &mut accelerations);

        for (idx, vels) in velocities.iter_mut().enumerate() {
            for k in 0..3 {
                vels[k] += 0.5 * (accelerations[idx][k] + old_accelerations[idx][k]) * time_step;
            }
        }
        let total_vel_squared: f64 = velocities.iter().flatten().map(|&x| x * x).sum();

        if time < NUM_TIME_STEPS / 2 && time % 5 == 0 {
            thermostat(&mut velocities);
        }

        if time > NUM_TIME_STEPS / 2 {
            let net_ke = 0.5 * MASS * total_vel_squared;

            ke.push(net_ke);
            pe.push(net_potential);
            total_e.push(net_ke + net_potential);
        }
    }
    let avg_pe = pe.iter().sum::<f64>() / pe.len() as f64;

    let sigma_over_l_over_two = SIGMA / (sim_length / 2.);
    let mut long_range_potential_corrections =
        (8.0 / 3.0) * PI as f64 * N as f64 * RHOSTAR * EPS_STAR;
    let temp = 1.0 / 3.0 * f64::powf(sigma_over_l_over_two, 9.);
    let temp1 = f64::powf(sigma_over_l_over_two, 3.);
    long_range_potential_corrections *= temp - temp1;
    let pestar = ((avg_pe + long_range_potential_corrections) / N as f64) / EPS_STAR;
    println!("Avg PE: {avg_pe}");
    println!("Average energy: {}", total_e[total_e.len() - 1]);
    println!("Reduced potential: {}", pestar);
}

fn calc_forces(
    positions: &[[f64; 3]; N as usize],
    accelerations: &mut [[f64; 3]; N as usize],
) -> f64 {
    let mut net_potential = 0.0;
    let sim_length = f64::cbrt(N as f64 / RHO);
    let cells_per_dimension = (f64::floor(sim_length / TARGET_CELL_LENGTH)) as i32;
    let cells_2d = cells_per_dimension.pow(2);
    let cells_3d = cells_per_dimension * cells_2d;
    let cell_length = sim_length / cells_per_dimension as f64;

    let mut header = vec![-1; cells_3d as usize];
    let mut cell_list = [0; N as usize];

    for atom_idx in 0..N {
        let x = (positions[atom_idx as usize][0] / cell_length) as i32;
        let y = (positions[atom_idx as usize][1] / cell_length) as i32;
        let z = (positions[atom_idx as usize][2] / cell_length) as i32;
        let c = x * cells_2d + y * cells_per_dimension + z;
        cell_list[atom_idx as usize] = header[c as usize];
        header[c as usize] = atom_idx;
    }

    for c in 0..cells_3d {
        net_potential +=
            calc_forces_on_cell(c as usize, accelerations, positions, &header, &cell_list);
    }

    net_potential
}

fn calc_forces_on_cell(
    cell_idx: usize,
    accel: &mut [[f64; 3]; N as usize],
    pos: &[[f64; 3]; N as usize],
    cell_header: &[i32],
    atom_cell_list: &[i32; N as usize],
) -> f64 {
    let mut potential = 0.0;
    let sim_length = f64::cbrt(N as f64 / RHO);
    let cells_per_dimension = f64::floor(sim_length / TARGET_CELL_LENGTH) as i32;
    let cells_2d = cells_per_dimension.pow(2);
    let cell_x = cell_idx as i32 / cells_2d;
    let remainder = cell_idx as i32 % cells_2d;
    let cell_y = remainder / cells_per_dimension;
    let cell_z = remainder % cells_per_dimension;

    for (one, two, three) in iproduct!(
        cell_x - 1..=cell_x + 1,
        cell_y - 1..=cell_y + 1,
        cell_z - 1..=cell_z + 1
    ) {
        let shifted_x = (one + cells_per_dimension) % cells_per_dimension;
        let shifted_y = (two + cells_per_dimension) % cells_per_dimension;
        let shifted_z = (three + cells_per_dimension) % cells_per_dimension;
        let neighbor_idx = shifted_x * cells_2d + shifted_y * cells_per_dimension + shifted_z;
        let mut i = cell_header[cell_idx];
        while i > -1 {
            let mut j = cell_header[neighbor_idx as usize];
            while j > -1 {
                if i < j {
                    let dist_arr: [f64; 3] = pos[i as usize]
                        .iter()
                        .zip(pos[j as usize].iter())
                        .map(|(&a, &b)| {
                            let diff = a - b;
                            diff - sim_length * f64::round(diff / sim_length)
                        })
                        .collect::<Vec<f64>>()
                        .try_into()
                        .unwrap();
                    let r2 = dot(&dist_arr);
                    assert_ne!(r2, 0.);
                    if r2 < R_CUTOFF_SQUARED {
                        let s2or2 = SIGMA * SIGMA / r2;
                        let sor6 = s2or2 * s2or2 * s2or2;
                        let sor12 = sor6 * sor6;
                        let force_over_r = 24. * EPS_STAR / r2 * (2. * sor12 - sor6);
                        potential += 4. * EPS_STAR * (sor12 - sor6);
                        for k in 0..3 {
                            accel[i as usize][k] += force_over_r * dist_arr[k] / MASS;
                            accel[j as usize][k] -= force_over_r * dist_arr[k] / MASS;
                        }
                    }
                }
                j = atom_cell_list[j as usize];
            }
            i = atom_cell_list[i as usize];
        }
    }

    potential
}

fn write_positions(pos: &[[f64; 3]], file: &mut BufWriter<File>, time: i32) {
    write!(file, "{}\nTime: {}\n", N, time).expect("File not found");
    for atom in pos.iter() {
        writeln!(file, "A {:.5} {:.5} {:.5}", atom[0], atom[1], atom[2]).expect("File not found");
    }
}

fn write_dbg(
    pos: &[[f64; 3]],
    vel: &[[f64; 3]],
    accel: &[[f64; 3]],
    old_accel: &[[f64; 3]],
    file: &mut BufWriter<File>,
    time: i32,
) {
    write!(file, "{}\nTime: {}\n", N, time).expect("File not found");
    for atom in pos.iter() {
        writeln!(file, "pos {:.5} {:.5} {:.5}", atom[0], atom[1], atom[2]).expect("File not found");
    }
    for atom in vel.iter() {
        writeln!(file, "vel {:.5} {:.5} {:.5}", atom[0], atom[1], atom[2]).expect("File not found");
    }
    for atom in accel.iter() {
        writeln!(file, "accel {:.5} {:.5} {:.5}", atom[0], atom[1], atom[2])
            .expect("File not found");
    }
    for atom in old_accel.iter() {
        writeln!(
            file,
            "old accel {:.5} {:.5} {:.5}",
            atom[0], atom[1], atom[2]
        )
        .expect("File not found");
    }
}

fn thermostat(velocities: &mut [[f64; 3]; N as usize]) {
    let mut instant_temp: f64 = velocities.iter().map(|&x| MASS * dot(&x)).sum();
    instant_temp /= (3 * N - 3) as f64;
    let temp_scalar = f64::sqrt(TARGET_TEMP / instant_temp);
    for vel in velocities.iter_mut().flatten() {
        *vel *= temp_scalar;
    }
}

#[inline]
fn dot(arr: &[f64; 3]) -> f64 {
    arr.iter().map(|&x| x * x).sum()
}

fn face_centered_cell() -> [[f64; 3]; N as usize] {
    let n: i32 = f64::cbrt(N as f64 / 4.) as i32;
    let sim_length = f64::cbrt(N as f64 / RHO);
    let dr: f64 = sim_length / n as f64;
    let dro2: f64 = dr / 2.0;

    let mut count = 0;
    let mut positions: [[f64; 3]; N as usize] = [[0.0; 3]; N as usize];

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                positions[count] = [i as f64 * dr, j as f64 * dr, k as f64 * dr];
                count += 1;
                positions[count] = [i as f64 * dr + dro2, j as f64 * dr + dro2, k as f64 * dr];
                count += 1;
                positions[count] = [i as f64 * dr + dro2, j as f64 * dr, k as f64 * dr + dro2];
                count += 1;
                positions[count] = [i as f64 * dr, j as f64 * dr + dro2, k as f64 * dr + dro2];
                count += 1;
            }
        }
    }
    positions
}
