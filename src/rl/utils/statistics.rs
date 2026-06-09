use std::collections::{BTreeMap, BTreeSet};

use plotters::prelude::*;

const COLORS: [&RGBColor; 6] = [&RED, &BLUE, &GREEN, &MAGENTA, &CYAN, &BLACK];
const N_BUCKETS: usize = 10;
const PLOT_WIDTH: u32 = 1024;
const PLOT_HEIGHT: u32 = 1024;

#[derive(Clone, Debug)]
pub struct StatPoint {
    pub episode: u32,
    pub completed: bool,
    pub values: Vec<(String, f32)>,
}

impl StatPoint {
    pub fn new(episode: u32, completed: bool, values: Vec<(String, f32)>) -> Self {
        Self {
            episode,
            completed,
            values,
        }
    }

    pub fn get(&self, name: &str) -> Option<f32> {
        self.values
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| *v)
    }
}

#[derive(Debug)]
struct Bucket {
    episode: u32,
    completed_rate: f32,
    values: Vec<(String, f32)>,
}

#[derive(Clone, Default)]
pub struct Statistics {
    pub history: Vec<StatPoint>,
}

impl Statistics {
    pub fn mean_statistics(all: Vec<Statistics>) -> Statistics {
        if all.is_empty() {
            return Statistics::default();
        }

        let n_points = all
            .iter()
            .map(|s| s.history.len())
            .min()
            .unwrap_or(0);

        if n_points == 0 {
            return Statistics::default();
        }

        let n_runs = all.len() as f32;
        let mut history = Vec::with_capacity(n_points);

        for i in 0..n_points {
            let episode = all[0].history[i].episode;

            let completed_rate = all
                .iter()
                .map(|s| s.history[i].completed as u32 as f32)
                .sum::<f32>()
                / n_runs;

            let mut keys = BTreeSet::new();
            for s in &all {
                for (k, _) in &s.history[i].values {
                    keys.insert(k.clone());
                }
            }

            let mut values = Vec::new();
            for key in keys {
                let mut sum = 0.0f32;
                let mut count = 0usize;

                for s in &all {
                    if let Some(v) = s.history[i].get(&key) {
                        sum += v;
                        count += 1;
                    }
                }

                if count > 0 {
                    values.push((key, sum / count as f32));
                }
            }

            history.push(StatPoint {
                episode,
                completed: completed_rate >= 0.5,
                values,
            });
        }

        Statistics { history }
    }

    pub fn push(&mut self, completed: bool, values: Vec<(String, f32)>) {
        let episode = self.history.len() as u32;
        self.history.push(StatPoint::new(episode, completed, values));
    }

    pub fn last_value(&self, metric: &str) -> Option<f32> {
        self.history
            .last()
            .and_then(|p| p.get(metric))
    }

    /// Retorna si l'últim episodi ha completat l'entorn.
    pub fn last_completed(&self) -> Option<bool> {
        self.history
            .last()
            .map(|p| p.completed)
    }

    pub fn add_metric_to_previous(
        &mut self,
        metric: impl Into<String>,
        value: f32,
    ) {
        let metric = metric.into();

        for point in self.history.iter_mut().rev() {
            // Stop once the metric already exists
            if point.get(&metric).is_some() {
                break;
            }

            point.values.push((metric.clone(), value));
        }
    }

    fn to_buckets(&self) -> Vec<Bucket> {
        let bucket_size = ((self.history.len() as f32) / (N_BUCKETS as f32)).ceil() as usize;
        let mut buckets = Vec::new();

        for chunk in self.history.chunks(bucket_size.max(1)) {
            let len = chunk.len() as f32;
            let episode = chunk.last().unwrap().episode;
            let completed_rate = chunk.iter().filter(|x| x.completed).count() as f32 / len;

            let mut keys = BTreeSet::new();
            for point in chunk {
                for (k, _) in &point.values {
                    keys.insert(k.clone());
                }
            }

            let mut values = Vec::new();
            for key in keys {
                let mut sum = 0.0f32;
                let mut count = 0usize;

                for point in chunk {
                    if let Some(v) = point.get(&key) {
                        sum += v;
                        count += 1;
                    }
                }

                if count > 0 {
                    values.push((key, sum / count as f32));
                }
            }

            buckets.push(Bucket {
                episode,
                completed_rate,
                values,
            });
        }

        buckets
    }

    fn metric_points(buckets: &[Bucket], metric: &str) -> Vec<(u32, f32)> {
        buckets
            .iter()
            .filter_map(|b| {
                b.values
                    .iter()
                    .find(|(k, _)| k == metric)
                    .map(|(_, v)| (b.episode, *v))
            })
            .collect()
    }

    pub fn plot(
        &self,
        path: &str,
        title: &str,
        series: &[(String, f32)],
        color_by_completion: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.history.is_empty() || series.is_empty() {
            println!("Empty statistics");
            return Ok(());
        }

        let buckets = self.to_buckets();
        if buckets.is_empty() {
            println!("Empty buckets");
            return Ok(());
        }

        let root = BitMapBackend::new(path, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut weights: Vec<f32> = series.iter().map(|(_, h)| h.max(0.0)).collect();
        if weights.iter().all(|w| *w <= 0.0) {
            weights = vec![1.0; series.len()];
        }

        let total_weight = weights.iter().sum::<f32>().max(f32::EPSILON);

        let mut remaining = root;
        let mut remaining_height = PLOT_HEIGHT;
        let mut remaining_weight = total_weight;
        let mut areas = Vec::with_capacity(series.len());

        for (idx, (_, weight)) in series.iter().enumerate() {
            let is_last = idx + 1 == series.len();

            let raw_h = if is_last {
                remaining_height
            } else {
                ((remaining_height as f32) * (*weight / remaining_weight)).round() as u32
            };

            let min_rest = (series.len() - idx - 1) as u32;
            let h = raw_h
                .max(1)
                .min(remaining_height.saturating_sub(min_rest).max(1));

            let (area, rest) = remaining.split_vertically(h);
            areas.push(area);
            remaining = rest;

            remaining_height = remaining_height.saturating_sub(h);
            remaining_weight -= *weight;
        }

        let x_min = buckets.first().unwrap().episode;
        let x_max = buckets.last().unwrap().episode.max(x_min + 1);

        for (idx, ((name, _), area)) in series.iter().zip(areas.iter()).enumerate() {
            let points = Self::metric_points(&buckets, name);

            if points.len() < 2 {
                continue;
            }

            let (mut y_min, mut y_max) = points.iter().fold(
                (f32::INFINITY, f32::NEG_INFINITY),
                |(mn, mx), (_, v)| (mn.min(*v), mx.max(*v)),
            );

            if (y_max - y_min).abs() < f32::EPSILON {
                y_min -= 1.0;
                y_max += 1.0;
            } else {
                let pad = (y_max - y_min) * 0.1;
                y_min -= pad;
                y_max += pad;
            }

            let mut chart = ChartBuilder::on(area)
                .caption(format!("{} — {}", title, name), ("sans-serif", 24))
                .margin(20)
                .x_label_area_size(40)
                .y_label_area_size(60)
                .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

            chart
                .configure_mesh()
                .x_desc("Episode")
                .y_desc(name)
                .draw()?;

            if color_by_completion {
                for pair in buckets.windows(2) {
                    let a = &pair[0];
                    let b = &pair[1];

                    let av = (a.completed_rate + b.completed_rate) / 2.0;
                    let color = if av >= 0.5 { &GREEN } else { &RED };

                    let ya = a.values.iter().find(|(k, _)| k == name).map(|(_, v)| *v);
                    let yb = b.values.iter().find(|(k, _)| k == name).map(|(_, v)| *v);

                    if let (Some(ya), Some(yb)) = (ya, yb) {
                        chart.draw_series(std::iter::once(PathElement::new(
                            vec![(a.episode, ya), (b.episode, yb)],
                            color.stroke_width(4),
                        )))?;
                    }
                }
            } else {
                let color = COLORS[idx % COLORS.len()];
                chart.draw_series(LineSeries::new(points, color.stroke_width(3)))?;
            }
        }

        remaining.present()?;
        Ok(())
    }

    pub fn plot_multiple(
        path: &str,
        title: &str,
        metric: &str,
        statistics_series: &[(String, Statistics)],
    ) -> Result<(), Box<dyn std::error::Error>> {
        if statistics_series.is_empty() {
            return Ok(());
        }

        let root = BitMapBackend::new(path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut all_points: Vec<(String, Vec<Bucket>)> = Vec::new();
        let mut x_min = u32::MAX;
        let mut x_max = 0u32;
        let mut any_points = false;

        for (label, stats) in statistics_series {
            let buckets = stats.to_buckets();
            if let Some(first) = buckets.first() {
                x_min = x_min.min(first.episode);
            }
            if let Some(last) = buckets.last() {
                x_max = x_max.max(last.episode);
            }

            if !Self::metric_points(&buckets, metric).is_empty() {
                any_points = true;
            }

            all_points.push((label.clone(), buckets));
        }

        if !any_points {
            println!("No points for metric '{}'", metric);
            return Ok(());
        }

        if x_min == u32::MAX {
            x_min = 0;
        }
        if x_max <= x_min {
            x_max = x_min + 1;
        }

        let mut y_min = f32::INFINITY;
        let mut y_max = f32::NEG_INFINITY;

        for (_, buckets) in &all_points {
            for (_, v) in Self::metric_points(buckets, metric) {
                y_min = y_min.min(v);
                y_max = y_max.max(v);
            }
        }

        if !y_min.is_finite() || !y_max.is_finite() {
            return Ok(());
        }

        if (y_max - y_min).abs() < f32::EPSILON {
            y_min -= 1.0;
            y_max += 1.0;
        } else {
            let pad = (y_max - y_min) * 0.1;
            y_min -= pad;
            y_max += pad;
        }

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 30))
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

        chart
            .configure_mesh()
            .x_desc("Episode")
            .y_desc(metric)
            .draw()?;

        for (idx, (label, buckets)) in all_points.iter().enumerate() {
            let color = COLORS[idx % COLORS.len()];
            let points = Self::metric_points(buckets, metric);

            if points.len() < 2 {
                continue;
            }

            chart
                .draw_series(LineSeries::new(points, color.stroke_width(3)))?
                .label(label.clone())
                .legend(move |(x, y)| {
                    PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(3))
                });
        }

        chart
            .configure_series_labels()
            .label_font(("sans-serif", 25))
            .position(SeriesLabelPosition::LowerRight)
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;

        root.present()?;
        Ok(())
    }
}