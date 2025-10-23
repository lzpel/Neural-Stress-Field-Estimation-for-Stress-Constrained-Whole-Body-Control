use std::{cmp::Ordering, fmt::Debug, ops::{Add, Div, Mul, Sub}};

#[derive(Default, Debug, Clone)]
struct Target {
	pub name: String,
	pub deps: Vec<String>,
}
trait HasTarget {
	fn target<'a>(&'a self) -> &'a Target;
	fn target_name(&self) -> &str {
		self.target().name.as_str()
	}
	fn target_finished(&self) -> bool;
	fn target_ready<'a>(&'a self, tasks: impl Iterator<Item = &'a Self> + Clone) -> bool
	where
		Self: Sized,
	{
		let v = &self.target().deps;
		v.iter().all(|v| {
			tasks
				.clone()
				.filter(|v| v.target_finished())
				.any(|x| x.target_name() == v)
		})
	}
}
trait Task<R: Default>: HasTarget + Clone {
	fn request_resource(&self) -> R {
		Default::default()
	}
	fn do_1sec(&mut self, r: &R) -> bool; //finish
}
#[derive(Default, Debug, Clone, Copy)]
struct Resource {
	network_gbps: f32, //Gbps
	vram_gbytes: f32,
}
trait Linear: Sized + Default + Debug {
	type V: Add<Output = Self::V>
		+ Sub<Output = Self::V>
		+ Mul<Output = Self::V>
		+ Div<Output = Self::V>
		+ PartialOrd
		+ Copy
		+ TryFrom<f32>
		+ Default; //算術系が大体使えるように詰め込む、minがしたい
	fn line(&self, rhs: &Self, f: impl Fn(Self::V, Self::V) -> Self::V) -> Self;
	fn add(&self, rhs: &Self) -> Self {
		self.line(&rhs, |a, b| a + b)
	}
	fn sub(&self, rhs: &Self) -> Self {
		self.line(&rhs, |a, b| a - b)
	}
	fn write(&self, w: &mut impl std::io::Write)->std::io::Result<()>;
}
impl Linear for Resource {
	type V = f32;
	fn line(&self, rhs: &Self, f: impl Fn(Self::V, Self::V) -> Self::V) -> Self {
		Self {
			vram_gbytes: f(self.vram_gbytes, rhs.vram_gbytes),
			network_gbps: f(self.network_gbps, rhs.network_gbps),
		}
	}
	fn write(&self, w: &mut impl std::io::Write) ->std::io::Result<()>{
		writeln!(w, "{} {}", self.vram_gbytes, self.network_gbps)
	}
}
#[derive(Debug, Clone)]
enum Any {
	Render(usize, f32),
	Compute(usize),
	Network(f32),
}
#[derive(Debug, Clone)]
struct AnyTask {
	any: Any,
	inner: Target,
}
impl HasTarget for AnyTask {
	fn target<'a>(&'a self) -> &'a Target {
		&self.inner
	}
	fn target_finished(&self) -> bool {
		match self.any {
			Any::Compute(t) => t <= Default::default(),
			Any::Render(t, _) => t <= Default::default(),
			Any::Network(v) => v <= Default::default(),
		}
	}
}
impl Task<Resource> for AnyTask {
	//エラー
	fn request_resource(&self) -> Resource {
		match self.any {
			Any::Render(_t, v) => Resource {
				vram_gbytes: v,
				..Default::default()
			},
			Any::Network(_v) => Resource {
				network_gbps: 1.,
				..Default::default()
			},
			_ => Default::default(),
		}
	}
	fn do_1sec(&mut self, assign: &Resource) -> bool {
		match &mut self.any {
			Any::Compute(t) => *t -= 1,
			Any::Render(t, _) => *t -= 1,
			Any::Network(v) => *v -= assign.network_gbps as f32 / 8.0,
		}
		Default::default()
	}
}
fn step<T: Task<R>, R: Linear>(
	mut write_res: impl std::io::Write,
	mut write_tas: impl std::io::Write,
	capacity: R,
	tasks: impl Iterator<Item = T>,
	strategy: impl Fn(&R, &R)->bool//budget resource, requested resource -> run(true) or pending(falce)
) -> std::io::Result<()> {
	let mut tasks: Vec<(T, [Option<usize>; 2])> = tasks.map(|v| (v, Default::default())).collect();
	//リソース上限をここに書く
	capacity.write(&mut write_res)?;
	for t in 0..usize::MAX {
		println!("step {t}");
		// 終了できるタスクは終了、全部終了していたら終わり
		let mut end = true;
		for (v, [a, b]) in &mut tasks {
			if [a.is_some(), b.is_some()] == [true, false] {
				if v.target_finished() {
					*b = Some(t);
					writeln!(write_tas, "{} {} {}", v.target_name(), a.unwrap(), b.unwrap())?;
				}
			}
			if b.is_none() {
				end = false;
			}
		}
		if end {
			break;
		}
		// 投入できるタスクがあれば投入
		while {
			let mut changed = false;
			// 現在要請中のリソースを合算
			let used = tasks
				.iter()
				.filter(|(v, [a, b])| [a.is_some(), b.is_some()] == [true, false])
				.map(|(v, _)| v.request_resource())
				.reduce(|a, b| a.add(&b))
				.unwrap_or_default();
			// 利用可能量を計算
			let budget = capacity.sub(&used);
			// 投入できるなら投
			for (i, (v, [a, b])) in tasks.iter().enumerate() {
				if [a.is_some(), b.is_some()] == [false, false] {
					if v.target_ready(tasks.iter().map(|(v, _)| v)) && strategy(&budget, &v.request_resource()) {
						tasks[i].1 = [Some(t), None];
						changed = true;
						break;
					}
				}
			}
			// ここで場合によっては利用可能量が負になっている。
			changed == true
		} {}
		// 投入中のリソースを合算
		let used = tasks
			.iter()
			.filter(|(v, [a, b])| [a.is_some(), b.is_some()] == [true, false])
			.map(|(v, _)| v.request_resource())
			.reduce(|a, b| a.add(&b))
			.unwrap_or_default();
		//超過分
		let one: <R as Linear>::V=match (1.).try_into(){
			Ok(v)=>v,
			Err(_)=>panic!()
		};
		let rate = used.line(&capacity, |a, b| if b!=Default::default() && a.partial_cmp(&b)==Some(Ordering::Greater) {a / b}else{one});
		used.write(&mut write_res)?;
		// 超過分を割り引いて渡しながらdo_1sec
		for (v, [a, b]) in &mut tasks {
			if [a.is_some(), b.is_some()] == [true, false] {
				let r = v.request_resource().line(&rate, |a, b| a / b);
				let _ = v.do_1sec(&r);
			}
		}
	}
	Ok(())
}
fn generate(faces: usize) -> Vec<AnyTask> {
	let mut tasks: Vec<AnyTask> = Default::default();
	let target_animation = "animation".to_string();
	tasks.push(AnyTask {
		any: Any::Compute(10),
		inner: Target {
			name: target_animation.clone(),
			..Default::default()
		},
	});
	for i in 0..faces {
		let target_render = format!("render{i}");
		tasks.push(AnyTask {
			any: Any::Render(100, 4.),
			inner: Target {
				name: target_render.clone(),
				deps: vec![target_animation.clone()],
			},
		});
		let target_archive = format!("archive{i}");
		tasks.push(AnyTask {
			any: Any::Compute(10),
			inner: Target {
				name: target_archive.clone(),
				deps: vec![target_render.clone()],
			},
		});
		let target_transfer = format!("transfer{i}");
		tasks.push(AnyTask {
			any: Any::Network((2 * 1024) as f32 / 2500 as f32),
			inner: Target {
				name: target_transfer.clone(),
				deps: vec![target_archive.clone()],
			},
		});
	}
	tasks
}
fn main() {
	let out_res = std::fs::File::create("out.res.csv").unwrap();
	let out_tas = std::fs::File::create("out.tas.csv").unwrap();
	let (task, capacity): (Vec<AnyTask>, Resource) = match 2{
		1=>(
			generate(2500),
			Resource {
				network_gbps: 10.,
				vram_gbytes: 48. * 8.,
			}
		),
		2=>(
			generate(20),
			Resource{
				network_gbps: 1.,
				vram_gbytes: 16.
			}
		),
		_ => panic!("unexpected mode"),
	};
	let fool_strategy=|budget: &Resource, request: &Resource|->bool{
		budget.vram_gbytes>=request.vram_gbytes
	};
	let smart_strategy=|budget: &Resource, request: &Resource|->bool{
		budget.vram_gbytes>=request.vram_gbytes && budget.network_gbps>=request.network_gbps
	};
	step(
		out_res,
		out_tas,
		capacity,
		task.into_iter(),
		smart_strategy,
	).unwrap();
}
