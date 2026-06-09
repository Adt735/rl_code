use std::{f32::consts::PI, process::Command};

use ::nalgebra::UnitQuaternion;
// use ::nalgebra::{UnitQuaternion, Vector3};
use rapier3d::{glamx::{Quat, Vec3Swizzles}, prelude::*};
use serde::{Deserialize, Serialize};
use plotters::prelude::*;
use tch::Tensor;

use crate::rl::{Action, EnvironmentTrait, State, ToTensor, environments::rapier_simulation::RapierSim};


const OBSTACLES_GROUP: InteractionGroups = InteractionGroups::new(Group::GROUP_1, Group::all(), InteractionTestMode::And);


#[derive(Clone)]
pub struct RayObservation2d {
    pub length: f32,
    pub angle: f32,
}


/**************************************************************
===============================================================
                        Environment
===============================================================
**************************************************************/
/// Wether the reward at each step is:
#[derive(Clone, Default, Debug, PartialEq, Eq, Deserialize, Copy)]
pub enum PlaneRewardType {
    #[default]
    /// -1.0
    Flat,
    /// Proportional to distance to goal
    Gradient,
}
impl PlaneRewardType {
    pub const ALL: [PlaneRewardType; 2] = [PlaneRewardType::Flat, PlaneRewardType::Gradient];
}

#[derive(Clone)]
pub struct TrajectoryPoints {
    pub position: Vec2,

    /// (angle, detection)
    pub rays: Vec<(f32, f32)>,
    pub rays_length: f32
}

pub struct Simple2dEnvironment {
    pub rays: Vec<RayObservation2d>,

    pub goal_pos: Vec2,
    pub initial_agent_pos: Vector,

    pub width: f32,
    pub height: f32,
    pub obstacles: Vec<Vec2>,

    pub simulation: Option<RapierSim>,
    pub agent_handle: Option<RigidBodyHandle>,
    pub agent_collider_handle: Option<ColliderHandle>,

    pub reward_type: PlaneRewardType,
    pub trajectory: Vec<TrajectoryPoints>
}
impl Simple2dEnvironment {
    pub fn new(
        width: f32, height: f32, agent_pos: Vec2, goal_pos: Vec2, obstacles: Vec<Vec2>, reward_type: PlaneRewardType,
        n_rays: usize, length_ray: f32, ray_span: f32
    ) -> Self {
        Self {
            rays: (0..n_rays).map(|v| RayObservation2d{ length: length_ray, angle: -ray_span/2.0 + v as f32 / (n_rays-1) as f32 * ray_span }).collect(),

            goal_pos,
            initial_agent_pos: Vector::new(agent_pos.x, 0.5, agent_pos.y),

            width,
            height,
            obstacles,

            simulation: None,
            agent_handle: None,
            agent_collider_handle: None,

            reward_type,
            trajectory: Vec::new(),
        }.generate_environment_sim()
    }

    fn generate_environment_sim(mut self) -> Self {
        let mut bodies = RigidBodySet::new();
        let mut colliders = ColliderSet::new();
        let center = vector![self.width/2.0, 0.0, self.height/2.0];

        // --- Ground (plane in XZ) ---
        let ground_body = RigidBodyBuilder::fixed()
            .translation(Vector::new(center.x, center.y, center.z))
            .build();
        let ground_handle = bodies.insert(ground_body);

        let ground_collider = ColliderBuilder::cuboid(
                self.width / 2.0,
                0.1,
                self.height / 2.0,
            )
            .build();
        colliders.insert_with_parent(ground_collider, ground_handle, &mut bodies);


        // --- Obstacles (cylinders) ---
        for pos in &self.obstacles {
            let body = RigidBodyBuilder::fixed()
                .translation(Vector::new(pos.x, 0.5, pos.y))
                .build();
            let handle = bodies.insert(body);

            let collider = ColliderBuilder::cylinder(0.5, 0.5)
                // .collision_groups(OBSTACLES_GROUP)
                .sensor(true)
                .active_events(ActiveEvents::COLLISION_EVENTS)
                .build();
            colliders.insert_with_parent(collider, handle, &mut bodies);
        }

        // --- Agent ---
        let agent_body = RigidBodyBuilder::kinematic_velocity_based()
            .translation(self.initial_agent_pos)
            .rotation(Vector::new(0.0, 0.0, 0.0))
            .build();
        let agent_handle = bodies.insert(agent_body);

        let agent_collider = ColliderBuilder::ball(0.25)
            .active_collision_types(
                ActiveCollisionTypes::default() | ActiveCollisionTypes::KINEMATIC_FIXED,
            )
            .build();
        let agent_collider_handle = colliders.insert_with_parent(agent_collider, agent_handle, &mut bodies);


        // Init sim
        let sim = RapierSim::new(bodies, colliders);
        self.simulation = Some(sim);
        self.agent_handle = Some(agent_handle);
        self.agent_collider_handle = Some(agent_collider_handle);
    
        self
    }

    fn get_rays_observation(&self) -> Vec<(f32, f32)> {
        let simulation = self.simulation.as_ref().unwrap();
        let agent = &simulation.bodies[self.agent_handle.unwrap()];

        // ---------- Agent position -------------
        let agent_pos = agent.position();
        let agent_translation = agent_pos.translation;
        let agent_rotation = agent_pos.rotation;

        self.rays.iter().map(|observation_ray| {
            let rays_rotation = (agent_rotation * Quat::from_axis_angle(Vector3::Y, observation_ray.angle)) * Vector3::Z;
            
            let ray = Ray::new(
                agent_translation, 
                rays_rotation,
            );
            let max_toi = observation_ray.length;
            let solid = true;
            let filter = QueryFilter::default().exclude_rigid_body(self.agent_handle.unwrap());

            let query_pipeline = simulation.broad_phase.as_query_pipeline(
                simulation.narrow_phase.query_dispatcher(),
                &simulation.bodies,
                &simulation.colliders,
                filter,
            );

            let detection = if let Some((_, toi)) = query_pipeline.cast_ray(
                &ray, max_toi, solid
            ) {
                toi / observation_ray.length
            } else {
                1.0
            };
            (rays_rotation.x.atan2(rays_rotation.z), detection)
        }).collect()
    }

    fn downsample_trajcetory(&self, max_points: usize) -> Vec<TrajectoryPoints> {
        let len = self.trajectory.len();

        if len <= max_points || max_points < 2 {
            return self.trajectory.clone();
        }

        let mut reduced = Vec::with_capacity(max_points);
        for i in 0..max_points {
            let fraction = i as f32 / (max_points - 1) as f32;
            let original_index = (fraction * (len - 1) as f32).round() as usize;
            
            reduced.push(self.trajectory[original_index].clone());
        }

        reduced
    }

    fn plot(&self, path: &str, until: Option<usize>, n_trajectory_points: usize) -> Result<(), Box<dyn std::error::Error>> {    
        let cell_size = 50;
        
        // ---------------------------------------------------------
        // Draw config
        // ---------------------------------------------------------
        let root = BitMapBackend::new(
            path,
            (
                (self.width * 1.5) as u32 * cell_size,
                (self.height * 1.5) as u32 * cell_size,
            ),
        )
        .into_drawing_area();
        root.fill(&WHITE)?;
    
        let mut chart = ChartBuilder::on(&root)
            .margin(0)
            .build_cartesian_2d(
                0f32..self.width as f32,
                0f32..self.height as f32,
            )?;
    
        chart
            .configure_mesh()
            // .disable_mesh()
            .x_labels(0)
            .y_labels(0)
            .draw()?;

        // ---------------------------------------------------------
        // Obstacles
        // ---------------------------------------------------------
        for obstacle in &self.obstacles {
            chart.draw_series(std::iter::once(Circle::new(
                (obstacle.x + 0.5, obstacle.y + 0.5),
                50,
                full_palette::GREY.filled(),
            )))?;
        }
    
        // ---------------------------------------------------------
        // Initial position
        // ---------------------------------------------------------
        chart
            .draw_series(std::iter::once(Circle::new(
                (
                    self.initial_agent_pos.x as f32 + 0.5,
                    self.initial_agent_pos.z as f32 + 0.5,
                ),
                10,
                RED,
            )))?
            .label("Start")
            .legend(|(x, y)| Circle::new((x, y), 5, RED.filled()));
    
        // ---------------------------------------------------------
        // Goal position
        // ---------------------------------------------------------
        chart
            .draw_series(std::iter::once(Circle::new(
                (
                    self.goal_pos.x as f32 + 0.5,
                    self.goal_pos.y as f32 + 0.5,
                ),
                20,
                GREEN.mix(0.5).filled(),
            )))?
            .label("Goal")
            .legend(|(x, y)| Circle::new((x, y), 5, GREEN.filled()));
    
        // ---------------------------------------------------------
        // Trajectory
        // ---------------------------------------------------------
        let mut trajectory = self.downsample_trajcetory(n_trajectory_points);
        if let Some(until) = until {
             let end = (until + 1).min(trajectory.len());
            trajectory = trajectory[..end].to_vec();
        }
        if trajectory.len() >= 2 { 
            let points: Vec<(f32, f32)> = trajectory
                .iter()
                .map(|p| {
                    (
                        p.position.x + 0.5,
                        p.position.y + 0.5,
                    )
                })
                .collect();
            
            for segment in points.windows(2) {
                chart.draw_series(std::iter::once(
                    PathElement::new(
                        vec![segment[0], segment[1]],
                        RED.mix(0.9).stroke_width(3),
                    )
                ))?;
            }
        }

        // ---------------------------------------------------------
        // Current position
        // ---------------------------------------------------------
        if until.is_some() {
            chart
                .draw_series(std::iter::once(Circle::new(
                    (
                        trajectory.last().unwrap().position.x as f32 + 0.5,
                        trajectory.last().unwrap().position.y as f32 + 0.5,
                    ),
                    8,
                    RED.filled(),
                )))?
                .label("Current pos")
                .legend(|(x, y)| Circle::new((x, y), 5, RED.filled()));
        }

        // ---------------------------------------------------------
        // Rays
        // --------------------------------------------------------
        if until.is_some() {
            let last_point = trajectory.last().unwrap();
            let origin = (
                last_point.position.x as f32 + 0.5,
                last_point.position.y as f32 + 0.5,
            );

            for &(angle, detection) in &last_point.rays {
                let length = detection * last_point.rays_length;

                let end = (
                    origin.0 + angle.sin() * length,
                    origin.1 + angle.cos() * length,
                );

                chart.draw_series(std::iter::once(
                    PathElement::new(
                        vec![origin, end],
                        BLUE.mix(0.9).stroke_width(1),
                    )
                ))?;
            }
        }


        root.present()?;
        Ok(())
    }
}
impl EnvironmentTrait<Environment2dState, Environment2dActions> for Simple2dEnvironment {
    fn get_state(&self) -> Environment2dState {
        let simulation = self.simulation.as_ref().unwrap();
        let agent = &simulation.bodies[self.agent_handle.unwrap()];

        // ---------- Raycasting -------------
        let rays_info = self.get_rays_observation();

        // ---------- Agent rotation -------------
        let rot = *agent.rotation();
        let forward = rot * Vector3::Z;
        let to_goal = Vector3::new(self.goal_pos.x, 0.5, self.goal_pos.y) - agent.translation();
        let f = Vector3::new(forward.x, 0.0, forward.z).normalize();
        let g = Vector3::new(to_goal.x, 0.0, to_goal.z).normalize();
        let cross = f.cross(g);
        let dot = f.dot(g);
        let angle = cross.y.atan2(dot); // radians

        Environment2dState {
            distance_to_goal: to_goal.xz().distance(Vector2::ZERO),
            angle_to_goal: angle,
            rays: rays_info.iter().map(|r| r.1).collect()
        }
    }

    fn step(&mut self, action: Environment2dActions) -> (f32, bool) {
        let simulation = self.simulation.as_mut().unwrap();
        let agent = simulation.bodies.get_mut(self.agent_handle.unwrap()).unwrap();
        agent.set_linvel(Vector3::ZERO, true);
        agent.set_angvel(Vector3::ZERO, true);

        let rot = *agent.rotation();
        let forward = rot * Vector3::Z;

        match action {
            Environment2dActions::Forward => agent.set_linvel(forward, true),
            Environment2dActions::TurnLeft => agent.set_angvel(Vector::new(0.0, -2.0*PI, 0.0), true),
            Environment2dActions::TurnRight => agent.set_angvel(Vector::new(0.0, 2.0*PI, 0.0), true),
        }

        self.simulation.as_mut().unwrap().step();

        // -------- Compute distance --------------
        let simulation = self.simulation.as_mut().unwrap();
        let agent = simulation.bodies.get_mut(self.agent_handle.unwrap()).unwrap();
        let goal_pos = Vector3::new(self.goal_pos.x, 0.5, self.goal_pos.y);
        let distance = agent.translation().xz().distance(goal_pos.xz());
        let agent_pos = agent.translation().xz();

        // -------- Compute collision with obstacles --------------
        let mut intersected = false;
        for (_, _, intersecting) in
            simulation.narrow_phase.intersection_pairs_with(self.agent_collider_handle.unwrap())
        {
            if intersecting {
                intersected = true;
                break;
            }
        }

        // ------------- Add trajectory ----------------------------
        let rays_info = self.get_rays_observation();
        self.trajectory.push(TrajectoryPoints {
            position: agent_pos,
            rays: rays_info,
            rays_length: self.rays.first().and_then(|r| Some(r.length)).unwrap_or(1.0),
        });
        
        (
            (-distance - if intersected { 50.0 } else { 0.0 } + if distance < 0.2 { 50.0 } else { 0.0 }) / (self.width*self.height), 
            distance < 0.2,
        )
    }

    fn reset(&mut self) {       
        let simulation = self.simulation.as_mut().unwrap();
        let agent = simulation.bodies.get_mut(self.agent_handle.unwrap()).unwrap();
        agent.set_translation(self.initial_agent_pos, true);
        agent.set_rotation(Quat::from_axis_angle(Vector3::Y, 0.0), true);
        self.trajectory.clear();
    }
    
    fn plot(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.plot(path, None, usize::MAX)
    }
    
    fn real_time_video(&self, frames_path: &str, video_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let n_frames = 60 * 10; //frames/s * s
        for i in 0..n_frames {
            self.plot(&format!("{frames_path}frame_{i:05}.png"), Some(i), n_frames)?;
        }

        let status = Command::new("ffmpeg")
            .args([
                "-loglevel", "error",
                "-hide_banner",
                "-y",
                "-framerate", "60",
                "-i", &format!("{frames_path}frame_%05d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                video_path,
            ])
            .status()?;
        if !status.success() {
            println!("Failed to convert to video")
        }

        Ok(())
    }
}

impl Clone for Simple2dEnvironment {
    fn clone(&self) -> Self {
        Self {
            rays: self.rays.iter().cloned().collect(),
            goal_pos: self.goal_pos.clone(), 
            initial_agent_pos: self.initial_agent_pos.clone(), 
            width: self.width.clone(), 
            height: self.height.clone(), 
            obstacles: self.obstacles.clone(), 
            simulation: None,
            agent_handle: None,
            agent_collider_handle: None,
            reward_type: self.reward_type.clone(),
            trajectory: self.trajectory.clone(),
        }.generate_environment_sim()
    }
}

/**************************************************************
===============================================================
                        Actions
===============================================================
**************************************************************/
#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug, Default, Serialize, Deserialize)]
#[repr(u8)]
pub enum Environment2dActions {
    #[default]
    Forward = 0,
    TurnLeft = 1,
    TurnRight = 2,
}
impl Action for Environment2dActions {
    fn get_all_actions() -> Vec<Self> {
        vec![Self::Forward, Self::TurnLeft, Self::TurnRight]
    }
}
impl ToTensor for Environment2dActions {
    fn len(&self) -> usize {
        1
    }

    fn to_vec(&self) -> Vec<f32> {
        vec![*self as u8 as f32]
    }
}

/**************************************************************
===============================================================
                        State
===============================================================
**************************************************************/
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct Environment2dState {
    pub distance_to_goal: f32,
    pub angle_to_goal: f32,
    
    /// Each contains a number between 0 and 1
    pub rays: Vec<f32>,
}
impl State for Environment2dState {}

impl ToTensor for Environment2dState {
    fn len(&self) -> usize {
        self.rays.len() + 2
    }

    fn to_vec(&self) -> Vec<f32> {
        let mut data: Vec<f32> = Vec::with_capacity(self.len());
        // data.extend(&[self.agent_pos.x, self.agent_pos.y]);
        data.push(self.distance_to_goal);
        data.push(self.angle_to_goal);
        data.extend(&self.rays);
        data
    }
}
