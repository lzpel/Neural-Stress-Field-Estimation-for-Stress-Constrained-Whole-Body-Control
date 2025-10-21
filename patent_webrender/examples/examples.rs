use std::ops::{Add, Sub, Mul, Div};

use rand::{rngs::StdRng, SeedableRng};

#[derive(Default, Debug, Clone)]
struct Target{
	pub name: String,
	pub deps: Vec<String>
}
trait HasTarget{
	fn target<'a>(&'a self)->&'a Target;
	fn target_name(&self)->&str{
		self.target().name.as_str()
	}
	fn target_finished(&self)->bool;
	fn target_ready<'a>(&'a self, tasks: impl Iterator<Item=&'a Self>+Clone)->bool
    where
        Self: Sized,{
		let v=&self.target().deps;
		v.iter().all(|v| tasks.clone().filter(|v| v.target_finished()).any(|x| x.target_name()==v))
	}
}
trait Task<R: Default>: HasTarget+Clone{
	fn request_resource(&self)->R{
		Default::default()
	}
	fn ready(&self, _capacity: &R)->bool{
		true //タスクを投入できるか判定、必要があれば書き換え
	}
	fn do_1sec(&mut self, r: &R)->bool;//finish
}
#[derive(Default, Debug, Clone, Copy)]
struct Resource{
	network_gbps: i32,//Gbps
	vram_gbytes: i32,
}
trait Linear: Sized+Default{
	type V:Add<Output=Self::V>+Sub<Output=Self::V>+Mul<Output=Self::V>+Div<Output=Self::V>+Ord+From<i32>+Default;//算術系が大体使えるように詰め込む、minがしたい
    fn line(&self, rhs: &Self, f: impl Fn(Self::V,Self::V)->Self::V) -> Self;
	fn add(&self, rhs: &Self)->Self{
		self.line(&rhs, |a,b| a+b)
	}
	fn sub(&self, rhs: &Self)->Self{
		self.line(&rhs, |a,b| a-b)
	}
}
impl Linear for Resource{
	type V=i32;
	fn line(&self, rhs: &Self, f: impl Fn(Self::V,Self::V)->Self::V)->Self{
		Self{
			vram_gbytes: f(self.vram_gbytes, rhs.vram_gbytes),
			network_gbps: f(self.network_gbps, rhs.network_gbps)
		}
	}
}
#[derive(Default, Debug, Clone)]
struct ComputeTask{
	seconds: usize,
	inner: Target
}
impl HasTarget for ComputeTask{
	fn target<'a>(&'a self)->&'a Target {
		&self.inner
	}
	fn target_finished(&self)->bool {
		self.seconds<=0
	}
}
impl Task<Resource> for ComputeTask{//エラー
	fn do_1sec(&mut self, _: &Resource)->bool {
		self.seconds-=1;
		Default::default()
	}
	fn request_resource(&self)->Resource{
		Default::default()
	}
	fn ready(&self, _capacity: &Resource)->bool{
		true //タスクを投入できるか判定、必要があれば書き換え
	}
}
#[derive(Default, Debug, Clone)]
struct RenderTask{
	pub vram: Resource,
	pub seconds: usize,
	inner: Target
}
impl HasTarget for RenderTask{
	fn target<'a>(&'a self)->&'a Target {
		&self.inner
	}
	fn target_finished(&self)->bool {
		self.seconds<=0
	}
}
impl Task<Resource> for RenderTask{//エラー
	fn request_resource(&self)->Resource {
		self.vram
	}
	fn ready(&self, assign: &Resource)->bool {
		assign.vram_gbytes>self.vram.vram_gbytes
	}
	fn do_1sec(&mut self, assign: &Resource)->bool {
		self.seconds-=1;
		Default::default()
	}
}
#[derive(Default, Debug, Clone)]
struct NetworkTask{
	pub gbytes: f32,
	inner: Target
}
impl HasTarget for NetworkTask{
	fn target<'a>(&'a self)->&'a Target {
		&self.inner
	}
	fn target_finished(&self)->bool {
		self.gbytes<=0.
	}
}
impl Task<Resource> for NetworkTask{//エラー
	fn request_resource(&self)->Resource {
		Resource { network_gbps: 1, ..Default::default() }
	}
	fn do_1sec(&mut self, assign: &Resource)->bool {
		self.gbytes-=assign.network_gbps as f32 / 8.0;
		Default::default()
	}
}
fn step<T: Task<R>, R: Linear>(capacity: R,tasks: impl Iterator<Item=dyn AsRef<T>>+Clone)->Vec<(String, [Option<usize>;2])>{
	let mut tasks:Vec<(T, [Option<usize>;2])>=tasks.map(|v| (v.as_ref().clone(), Default::default())).collect();
	for i in 0..usize::MAX{
		// 終了できるタスクは終了、全部終了していたら終わり
		let mut end=true;
		for (v,[a,b]) in &mut tasks{
			if [a.is_some(),b.is_some()]==[true, false]{
				if v.target_finished(){
					*b=Some(i)
				}
			}
			if b.is_none(){
				end=false;
			}
		}
		if end{
			break;
		}
		// 投入できるタスクがあれば投入
		while{
			let mut changed=false;
			// 現在要請中のリソースを合算
			let used=tasks
				.iter()
				.filter(|(v,[a,b])| [a.is_some(),b.is_some()]==[true,false])
				.map(|(v,_)| v.request_resource())
				.reduce(|a,b| a.add(&b))
				.unwrap_or_default();
			// 利用可能量を計算
			let available=capacity.sub(&used);
			// 投入できるなら投入
			for i in 0..tasks.len(){
				let (v,[a,b])=&tasks[i];
				if [a.is_some(),b.is_some()]==[false, false]{
					if v.target_ready(tasks.iter().map(|(v,_)| v)) && v.ready(&available){
						tasks[i].1[0]=Some(i);
						changed=true;
						break;
					}
				}
			}
			// ここで場合によっては利用可能量が負になっている。
			changed==true
		}{}
		// 投入中のリソースを合算
		let used=tasks
			.iter()
			.filter(|(v,[a,b])| [a.is_some(),b.is_some()]==[true,false])
			.map(|(v,_)| v.request_resource())
			.reduce(|a,b| a.add(&b))
			.unwrap_or_default();
		//超過分
		let max=100;
		let over=used.line(&capacity, |a,b| (a/b*max.into()).min(max.into()));
		// 超過分を割り引いて渡しながらdo_1sec
		for (v,[a,b]) in &mut tasks{
			if [a.is_some(),b.is_some()]==[true,false]{
				let r=v.request_resource().line(&over, |a,b| a*max.into()/b);
				let _=v.do_1sec(&r);
			}
		}
	}
	tasks.into_iter().map(|(v, ab)| (v.target_name().to_string(), ab)).collect()
}
fn generate(faces: usize)->Vec<Box<dyn Task<Resource>>>{
	let mut rng = StdRng::seed_from_u64(faces as u64);
	let mut tasks: Vec<Box<dyn Task<Resource>>> = Default::default();
	let target_animation="animation".to_string();
	tasks.push(Box::new(ComputeTask{
		seconds: 10,
		inner: Target { 
			name: target_animation.clone(), 
			..Default::default()
		}
	}));
	for i in 0..faces{
		let target_render=format!("render{i}");
		tasks.push(Box::new(RenderTask{
			seconds: 100,
			vram: Resource { 
				vram_gbytes: 4,
				..Default::default()
			},
			inner: Target {
				name: target_render.clone(),
				deps: vec![target_animation.clone()]
			}
		}));
		let target_archive=format!("archive{i}");
		tasks.push(Box::new(RenderTask{
			seconds: 100,
			vram: Resource { 
				vram_gbytes: 4,
				..Default::default()
			},
			inner: Target {
				name: target_archive.clone(),
				deps: vec![target_render.clone()]
			}
		}));
		let target_transfer=format!("transfer{i}");
		tasks.push(Box::new(RenderTask{
			seconds: 100,
			vram: Resource { 
				vram_gbytes: 4,
				..Default::default()
			},
			inner: Target {
				name: target_transfer.clone(),
				deps: vec![target_archive.clone()]
			}
		}));
	}
	tasks
}
fn main(){
	let tasks=generate(10);
	let output = step(Resource{
		network_gbps: 10,
		vram_gbytes: 48*8,
	}, tasks.into_iter());
}