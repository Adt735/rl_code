use std::{fs::File, time, io::Write};

use plotters::prelude::*;

use crate::rl::utils::prelude::count_files;


const COLORS: [&RGBColor; 6] = [&RED, &BLUE, &GREEN, &MAGENTA, &CYAN, &BLACK];
const N_BUCKETS: usize = 10;

#[derive(Debug)]
struct Bucket {
    episode: u32,
    reward: f32,
    elapsed: f32,
    info: f32,
    success_rate: f32,
}


#[derive(Clone)]
pub struct StatPoint {
    pub episode: u32,
    pub reward_sum: f32,
    pub success: bool,
    pub elapsed: f32,
    pub info: f32,
}


#[derive(Clone, Default)]
pub struct Statistics {
    pub start_time: Option<time::Instant>,
    pub history: Vec<StatPoint>,
}
impl Statistics {
    pub fn mean_statistics(all: Vec<Statistics>) -> Statistics {
        if all.is_empty() {
            return Statistics::default();
        }

        let n_runs = all.len() as f32;
        let n_points = all[0].history.len();

        let mut history = Vec::with_capacity(n_points);

        for i in 0..n_points {
            let episode = all[0].history[i].episode;

            let reward_sum = all
                .iter()
                .map(|s| s.history[i].reward_sum)
                .sum::<f32>()
                / n_runs;

            let success_rate = all
                .iter()
                .map(|s| s.history[i].success as u32 as f32)
                .sum::<f32>()
                / n_runs;

            let elapsed = all
                .iter()
                .map(|s| s.history[i].elapsed)
                .sum::<f32>()
                / n_runs;

            let info = all
                .iter()
                .map(|s| s.history[i].info)
                .sum::<f32>()
                / n_runs;

            history.push(StatPoint {
                episode,
                reward_sum,
                success: success_rate >= 0.5,
                elapsed,
                info,
            });
        }

        Statistics {
            start_time: None,
            history,
        }
    }

    pub fn push(&mut self, rewards: &[f32], success: bool, info: f32) {
        let episode = self.history.len();
        if self.start_time.is_none() { self.start_time = Some(time::Instant::now()) }

        self.history.push(StatPoint { 
            episode: episode as u32,
            reward_sum: rewards.iter().sum::<f32>(),
            success, 
            elapsed: self.start_time.unwrap().elapsed().as_secs_f32(), 
            info,
        });
    }

    fn to_buckets(&self) -> Vec<Bucket> {
        let bucket_size =
            ((self.history.len() as f32) / (N_BUCKETS as f32))
                .ceil() as usize;

        let mut buckets = Vec::new();

        for chunk in self.history.chunks(bucket_size.max(1)) {
            let len = chunk.len() as f32;

            buckets.push(Bucket {
                episode: chunk.last().unwrap().episode,

                reward: chunk
                    .iter()
                    .map(|x| x.reward_sum)
                    .sum::<f32>()
                    / len,

                elapsed: chunk
                    .iter()
                    .map(|x| x.elapsed)
                    .sum::<f32>()
                    / len,

                info: chunk
                    .iter()
                    .map(|x| x.info)
                    .sum::<f32>()
                    / len,

                success_rate: chunk
                    .iter()
                    .filter(|x| x.success)
                    .count() as f32
                    / len as f32,
            });
        }

        buckets
    }

    pub fn plot(
        &self,
        path: &str,
        title: &str,
        info_name: &str,
        show_reward: bool,
        show_elapsed: bool,
        show_info: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.history.is_empty() {
            println!("Empty statistics");
            return Ok(());
        }

        let buckets = self.to_buckets();

        // -------------------- Plot size --------------------------------------
        let reward_height = if show_reward { 768 } else { 0 };
        let elapsed_height = if show_elapsed { 256 } else { 0 };
        let info_height = if show_info { 256 } else { 0 };

        let total_height =
            reward_height +
            elapsed_height +
            info_height;

        let root = BitMapBackend::new(path, (1024, total_height))
            .into_drawing_area();

        root.fill(&WHITE)?;

        let mut areas = Vec::new();

        let mut remaining = root;

        if show_reward {
            let (reward_area, rest) =
                remaining.split_vertically(reward_height);

            areas.push(reward_area);
            remaining = rest;
        }

        if show_elapsed {
            let (elapsed_area, rest) =
                remaining.split_vertically(elapsed_height);

            areas.push(elapsed_area);
            remaining = rest;
        }

        if show_info {
            let (info_area, _) =
                remaining.split_vertically(info_height);

            areas.push(info_area);
        }

        let x_min = buckets.first().unwrap().episode;
        let x_max = buckets.last().unwrap().episode;

        let reward_min = buckets
            .iter()
            .map(|b| b.reward)
            .fold(f32::INFINITY, f32::min);

        let reward_max = buckets
            .iter()
            .map(|b| b.reward)
            .fold(f32::NEG_INFINITY, f32::max);

        let mut area_idx = 0;

        //
        // Reward Plot
        //
        if show_reward {
            let area = &areas[area_idx];
            area_idx += 1;

            let mut chart = ChartBuilder::on(area)
                .caption(title, ("sans-serif", 30))
                .margin(20)
                .x_label_area_size(40)
                .y_label_area_size(60)
                .build_cartesian_2d(
                    x_min..x_max,
                    reward_min..reward_max,
                )?;

            chart
                .configure_mesh()
                .x_desc("Episode")
                .y_desc("Reward")
                .draw()?;

            for pair in buckets.windows(2) {
                let a = &pair[0];
                let b = &pair[1];

                let color = if (a.success_rate + b.success_rate) / 2.0 > 0.5 {
                    &GREEN
                } else {
                    &RED
                };

                chart.draw_series(std::iter::once(
                    PathElement::new(
                        vec![
                            (a.episode, a.reward),
                            (b.episode, b.reward),
                        ],
                        color.stroke_width(4),
                    ),
                ))?;
            }
        }

        //
        // Elapsed Plot
        //
        if show_elapsed {
            let elapsed_min = buckets
                .iter()
                .map(|b| b.elapsed)
                .fold(f32::INFINITY, f32::min);

            let elapsed_max = buckets
                .iter()
                .map(|b| b.elapsed)
                .fold(f32::NEG_INFINITY, f32::max);

            let area = &areas[area_idx];
            area_idx += 1;

            let mut chart = ChartBuilder::on(area)
                .caption("Elapsed Time", ("sans-serif", 25))
                .margin(20)
                .x_label_area_size(40)
                .y_label_area_size(60)
                .build_cartesian_2d(
                    x_min..x_max,
                    elapsed_min..elapsed_max,
                )?;

            chart
                .configure_mesh()
                .x_desc("Episode")
                .y_desc("Seconds")
                .draw()?;

            chart.draw_series(LineSeries::new(
                buckets.iter().map(|b| {
                    (b.episode, b.elapsed)
                }),
                &BLUE,
            ))?;
        }

        //
        // Info Plot
        //
        if show_info {
            let info_min = buckets
                .iter()
                .map(|b| b.info)
                .fold(f32::INFINITY, f32::min);

            let info_max = buckets
                .iter()
                .map(|b| b.info)
                .fold(f32::NEG_INFINITY, f32::max);

            let area = &areas[area_idx];

            let mut chart = ChartBuilder::on(area)
                .caption(info_name, ("sans-serif", 25))
                .margin(20)
                .x_label_area_size(40)
                .y_label_area_size(60)
                .build_cartesian_2d(
                    x_min..x_max,
                    info_min..info_max,
                )?;

            chart
                .configure_mesh()
                .x_desc("Episode")
                .y_desc(info_name)
                .draw()?;

            chart.draw_series(LineSeries::new(
                buckets.iter().map(|b| {
                    (b.episode, b.info)
                }),
                &MAGENTA,
            ))?;
        }

        remaining.present()?;

        Ok(())
    }

    pub fn plot_multiple(path: &str, title: &str, statistics_series: &[(String, Statistics)]) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        let buckets = statistics_series.iter().cloned().map(|(l, s)| (l, s.to_buckets())).collect::<Vec<_>>();
        let buckets_flat = buckets.iter().map(|(l, b)| b).flatten().collect::<Vec<_>>();
        let x_min = buckets_flat.first().unwrap().episode;
        let x_max = buckets_flat.last().unwrap().episode;

        let reward_min = buckets_flat
            .iter()
            .map(|b| b.reward)
            .fold(f32::INFINITY, f32::min);
        let reward_max = buckets_flat
            .iter()
            .map(|b| b.reward)
            .fold(f32::NEG_INFINITY, f32::max);


        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 30))
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(
                x_min..x_max,
                reward_min..reward_max,
            )?;
        chart
            .configure_mesh()
            .x_desc("Episode")
            .y_desc("Reward")
            .draw()?;

        
        for (idx, (label, statistics)) in statistics_series.iter().enumerate() {
            let color = COLORS[idx % COLORS.len()];
            let buckets = statistics.to_buckets();

            let points: Vec<(u32, f32)> = buckets
                .iter()
                .map(|b| (b.episode, b.reward))
                .collect();

            chart
                .draw_series(LineSeries::new(points, color.stroke_width(3)))?
                .label(label.clone())
                .legend(move |(x, y)| {
                    PathElement::new(
                        vec![(x, y), (x + 20, y)],
                        color.stroke_width(3),
                    )
                });
        }

        chart
            .configure_series_labels()
            .label_font(("sans-serif", 25))
            .position(SeriesLabelPosition::LowerRight)
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;

        Ok(())
    }
}
