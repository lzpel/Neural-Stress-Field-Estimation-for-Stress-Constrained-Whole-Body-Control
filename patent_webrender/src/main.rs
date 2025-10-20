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
					0=>TaskNew::NetDownloadTask(rng.random_range(0..3)),
					1=>TaskNew::NetUploadTask(rng.random_range(0..3)),
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
#[derive(Default, Debug)]
struct Resource{
	network_upload_gbps: usize,//Gbps
	network_download_gbps: usize,//Gbps
	vram_gbytes: usize,
}
enum TaskNew{
	ComputeTask,
	RenderTask(usize),
	NetUploadTask(usize),
	NetDownloadTask(usize),
}
#[derive(Default, Debug)]
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
			TaskNew::NetUploadTask(i)=>{
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
			TaskNew::NetDownloadTask(i)=>{
				Self{
					name: "download".to_string(),
					seconds,
					resource:Resource{
						network_download_gbps:i,
						..Default::default()
					},
					..Default::default()
				}
			}
		}
	}
}
struct Server{
	resource: Resource,
}
impl Server{
	pub fn simulate<'a>(&self, x: impl Iterator<Item=&'a Task>){
		let mut total=0;
		for (i,x) in x.enumerate(){
			println!("{i}:\t{x:?}");
			total+=x.seconds
		}
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
