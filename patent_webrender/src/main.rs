use std::{
	fs, io,
	ops::{Add, Mul, Sub},
	usize,
};

use rand::prelude::*;
struct Job {
	tasks: Vec<Task>,
}
impl Job {
	fn new(count: usize) -> Self {
		let mut rng = StdRng::seed_from_u64(count as u64);
		// 3. 0から10000までの乱数を10個生成
		let random_numbers: Vec<Task> = (0..count)
			.map(|_| match rng.random_range(0..3) {
				1 => Task::new(
					rng.random_range(4..5),
					TaskNew::UploadTask(rng.random_range(1..3)),
				),
				2 => Task::new(
					rng.random_range(200..300),
					TaskNew::RenderTask(rng.random_range(0..10)),
				),
				_ => Task::new(rng.random_range(10..50), TaskNew::ComputeTask),
			})
			.collect();
		Self {
			tasks: random_numbers,
		}
	}
}
#[derive(Default, Debug, Clone)]
struct Resource {
	network_upload_gbps: i32,   //Gbps
	network_download_gbps: i32, //Gbps
	vram_gbytes: i32,
}
trait AsVector: Sized {
	type Vector;
	fn as_vector(&self) -> Self::Vector;
	fn from_vector(v: &Self::Vector) -> Self;
	// このメソッドだけ「Vector = [E; N]」であることを要求
	fn add<E: Add<Output = E> + Mul<Output = E> + Copy, const N: usize>(
		&self,
		rhs: &Self,
		scalar: E,
	) -> Self
	where
		Self: AsVector<Vector = [E; N]>,
	{
		let a: [E; N] = self.as_vector();
		let b: [E; N] = rhs.as_vector();
		let out = std::array::from_fn(|i| a[i] + b[i] * scalar);
		Self::from_vector(&out) // equality bound により型が一致
	}
}
impl AsVector for Resource {
	type Vector = [i32; 3];
	fn as_vector(&self) -> Self::Vector {
		[
			self.vram_gbytes,
			self.network_download_gbps,
			self.network_upload_gbps,
		]
	}
	fn from_vector(x: &Self::Vector) -> Self {
		Self {
			network_upload_gbps: x[2],
			network_download_gbps: x[1],
			vram_gbytes: x[0],
		}
	}
}
impl Add for Resource {
	type Output = Self;
	fn add(self, rhs: Self) -> Self {
		AsVector::add(&self, &rhs, 1)
	}
}
impl Sub for Resource {
	type Output = Self;
	fn sub(self, rhs: Self) -> Self {
		AsVector::add(&self, &rhs, -1)
	}
}
enum TaskNew {
	ComputeTask,
	RenderTask(i32),
	UploadTask(i32),
}
#[derive(Default, Debug, Clone)]
struct Task {
	resource: Resource,
	name: String,
	time_in: Option<usize>,
	time_span: usize,
}
impl Task {
	fn new(time_span: usize, x: TaskNew) -> Task {
		match x {
			TaskNew::ComputeTask => Self {
				name: "compute".to_string(),
				time_span,
				..Default::default()
			},
			TaskNew::RenderTask(i) => Self {
				name: "render".to_string(),
				time_span,
				resource: Resource {
					vram_gbytes: i,
					..Default::default()
				},
				..Default::default()
			},
			TaskNew::UploadTask(i) => Self {
				name: "upload".to_string(),
				time_span,
				resource: Resource {
					network_upload_gbps: i,
					..Default::default()
				},
				..Default::default()
			},
		}
	}
	fn run(&mut self, time: usize) {
		if self.is_pending() {
			self.time_in = Some(time);
		} else {
			panic!()
		}
	}
	fn is_running(&self, time: usize) -> bool {
		self.time_in
			.is_some_and(|v| v <= time && time < v + self.time_span)
	}
	fn is_pending(&self) -> bool {
		self.time_in.is_none()
	}
	fn is_finished(&self, time: usize) -> bool {
		self.time_in.is_some() && !self.is_running(time)
	}
}
struct Server {
	resource: Resource,
}
impl Server {
	pub fn simulate<'a>(&self, x: impl Iterator<Item = &'a Task>) -> Vec<Task> {
		let mut v: Vec<Task> = x.cloned().collect();
		// 時間
		let mut t: usize = 0;
		loop {
			// 投入できるリソースがあれば投入
			while {
				let mut changed = false;
				// 現在使用中のリソースを合算
				let used = v
					.iter()
					.filter_map(|v| v.is_running(t).then_some(v.resource.clone()))
					.reduce(|a, b| a + b)
					.unwrap_or_default();
				let capacity = self.resource.clone() - used;
				println!("at {t} capacity {:?}", &capacity);
				// 投入できるタスクを投入
				for i in v.iter_mut().filter(|v| v.is_pending()) {
					if (capacity.clone() - i.resource.clone())
						.as_vector()
						.iter()
						.all(|v| *v >= 0)
					{
						i.run(t);
						println!("at {t} start {} {:?}", i.name, i.clone());
						changed = true;
						break;
					}
				}
				changed
			} {}
			// 最初に終わる時刻
			let t1 = v
				.iter()
				.filter(|v| v.is_running(t))
				.map(|i| i.time_span + i.time_in.unwrap())
				.reduce(|a, b| a.min(b))
				.unwrap_or(usize::MAX);
			println!("{t}->{t1}");
			t = t1;
			if v.iter().all(|v| v.is_finished(t)) {
				break;
			}
			println!("end t={t}");
		}
		return v;
	}
	pub fn visualize<'a>(
		&self,
		mut w: impl io::Write,
		x: impl Iterator<Item = &'a Task> + Clone,
	) -> io::Result<()> {
		for i in x.clone().enumerate() {
			let t = i.1.time_in.unwrap();
			let mut stage = 0;
			for j in x.clone().enumerate().filter(|v| v.0 < i.0) {
				if j.1.is_running(t) {
					stage += 1;
				}
			}
			let used = x
				.clone()
				.filter_map(|v| v.is_running(t).then_some(v.resource.clone()))
				.reduce(|a, b| a + b)
				.unwrap_or_default();
			writeln!(
				w,
				"{t} {} {} {} {} {}",
				i.1.time_span, used.vram_gbytes, used.network_upload_gbps, stage, i.1.name
			)?
		}
		Ok(())
	}
}
fn main() {
	let x = Job::new(2500);
	let s = Server {
		resource: Resource {
			network_upload_gbps: 10,
			network_download_gbps: 10,
			vram_gbytes: 48 * 8,
		},
	};
	let output = s.simulate(x.tasks.iter());
	let file = fs::File::create("out.csv").unwrap();
	s.visualize(file, output.iter()).unwrap();
}
