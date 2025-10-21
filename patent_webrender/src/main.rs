use std::ops::Add;

use rand::prelude::*;
struct Job{
	tasks: Vec<Task>
}
impl Job{
	fn new(count: usize)->Self{
		let mut rng = StdRng::seed_from_u64(count as u64);
		// 3. 0から10000までの乱数を10個生成
		let random_numbers: Vec<Task> = (0..count)
			.map(|_| {
				let seconds=rng.random_range(0..=100) ;
				let w=match rng.random_range(0..3){
					1=>TaskNew::UploadTask(rng.random_range(1..3)),
					2=>TaskNew::RenderTask(rng.random_range(0..10)),
					_=>TaskNew::ComputeTask,
				};
				Task::new(seconds, w)
			})
			.collect();
		Self{
			tasks: random_numbers
		}
	}
}
#[derive(Default, Debug, Clone)]
struct Resource{
	network_upload_gbps: usize,//Gbps
	network_download_gbps: usize,//Gbps
	vram_gbytes: usize,
}
trait AsVector: Sized {
    type Vector;
    fn as_vector(&self) -> Self::Vector;
    fn from_vector(v: &Self::Vector) -> Self;
    // このメソッドだけ「Vector = [E; N]」であることを要求
    fn add<E: Add<Output = E> + Copy, const N: usize>(&self, rhs: &Self) -> Self
    where
        Self: AsVector<Vector = [E; N]>,
    {
        let a: [E; N] = self.as_vector();
        let b: [E; N] = rhs.as_vector();
        let out = std::array::from_fn(|i| a[i] + b[i]);
        Self::from_vector(&out) // equality bound により型が一致
    }
}
impl AsVector for Resource{
	type Vector = [usize; 3];
	fn as_vector(&self) -> Self::Vector{
		[self.vram_gbytes, self.network_download_gbps, self.network_upload_gbps]
	}
	fn from_vector(x: &Self::Vector)->Self{
		Self { network_upload_gbps: x[2], network_download_gbps: x[1], vram_gbytes: x[0] }
	}
}
impl Add for Resource {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
		AsVector::add(&self, &rhs)
    }
}
enum TaskNew{
	ComputeTask,
	RenderTask(usize),
	UploadTask(usize),
}
#[derive(Default, Debug, Clone)]
struct Task{
	resource: Resource,
	name: String,
	seconds: usize
}
impl Task{
	fn new(seconds: usize, x: TaskNew)->Task{
		match x{
			TaskNew::ComputeTask=>{
				Self{
					name: "compute".to_string(),
					..Default::default()
				}
			},
			TaskNew::RenderTask(i)=>{
				Self{
					name: "render".to_string(),
					seconds,
					resource:Resource{
						vram_gbytes:i,
						..Default::default()
					},
					..Default::default()
				}
			},
			TaskNew::UploadTask(i)=>{
				Self{
					name: "upload".to_string(),
					seconds,
					resource:Resource{
						network_upload_gbps:i,
						..Default::default()
					},
					..Default::default()
				}
			},
		}
	}
}
struct Server{
	resource: Resource,
}
impl Server{
	pub fn simulate<'a>(&self, x: impl Iterator<Item=&'a Task>){
		let mut v: Vec<(i32, Task)>=x.map(|v| (-1,v.clone())).collect();
		// 時間
		let mut t: i32=0;
		// 現在使用中のリソースを合算
		let used: Resource = Default::default();
		for x in v.iter(){
			println!("total {total}")
		}
}
fn main() {
	let x=Job::new(100);
	let s=Server{
		resource: Resource { 
			network_upload_gbps: 10,
			network_download_gbps: 10,
			vram_gbytes: 48*8
		}
	};
	s.simulate(x.tasks.iter());
}
