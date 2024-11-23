import justpy as jp

import numpy as np
import matplotlib.pyplot as plt

import json

import sys
sys.path.append("../")

def pose3d_visualize(ax, motion, scores, elivation, angle, keypoints_threshold=0.7):
    joint_pairs = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]]
    joint_pairs_left = [[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]]
    joint_pairs_right = [[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]]

    color_mid = "#00457E"
    color_left = "#02315E"
    color_right = "#2F70AF"

    j3d = motion[:,:,0]
    ax.set_xlim(-512, 0)
    ax.set_ylim(-256, 256)
    ax.set_zlim(-512, 0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=elivation, azim=angle)
    plt.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
    for i in range(len(joint_pairs)):
        if scores[0][i] < keypoints_threshold:
            continue
        limb = joint_pairs[i]
        xs, ys, zs = [np.array([j3d[limb[0], j], j3d[limb[1], j]]) for j in range(3)]
        if joint_pairs[i] in joint_pairs_left:
            ax.plot(-xs, -zs, -ys, color=color_left, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
        elif joint_pairs[i] in joint_pairs_right:
            ax.plot(-xs, -zs, -ys, color=color_right, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
        else:
            ax.plot(-xs, -zs, -ys, color=color_mid, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization

def result_view(node_dict: dict):
    # 웹캠 이미지 표시할 컨테이너 설정
    result_view_container = jp.Div(
        a           = node_dict["plane"], 
        id          = "result_view_container", 
        classes     = "flex border-box justify-evenly items-end bg-white"
    )
    node_dict["result_view_container"] = result_view_container

    # ======================= Image View =======================
    img_view_container = jp.Div(
        a           = result_view_container,
        id          = "img_view_container",
        classes     = "flex flex-col w-1/3 border-box justify-center items-center bg-white"
    )
    
    img_plane = jp.Img(
        a           = img_view_container,
        id          = "webcam_img",
        classes     = "block w-full max-w-3xl",
        width       = "640",
        height      = "480"
    )
    node_dict["img_plane"] = img_plane

    frame_slider_info = jp.Span(
        a           = img_view_container,
        id          = "frame_slider_info",
        text        = "Frame 0/0",
        classes     = "text-lg"
    )
    
    frame_slider = jp.Input(
        a           = img_view_container,
        id          = "frame_slider",
        type        = "range",
        min         = "0",
        max         = "99",
        value       = "0",
        step        = "1",
        classes     = "w-full"
    )

    # ======================= Pose 3D View =======================
    pose3d_figures_view_container = jp.Div(
        a           = result_view_container,
        id          = "pose3d_figures_view_container",
        classes     = "flex flex-col w-1/3 border-box justify-between items-center bg-white"
    )
    
    pose3d_figures = jp.Matplotlib(
        a           = pose3d_figures_view_container,
        id          = "pose3d_figures",
        classes     = "w-full max-h-80 mb-10 flex justify-center items-center"
    )

    elivation_slider_info = jp.Span(
        a           = pose3d_figures_view_container,
        text        = "Elevation 90/180deg",
        classes     = "text-lg mt-5"
    )
    
    elivation_slider = jp.Input(
        a           = pose3d_figures_view_container,
        id          = "elivation_slider",
        type        = "range",
        min         = "0",
        max         = "179",
        value       = "90",
        step        = "0.01",
        classes     = "w-full"
    )
    
    rotation_slider_info = jp.Span(
        a           = pose3d_figures_view_container,
        text        = "Rotation 0/360deg",
        classes     = "text-lg mt-5"
    )
    
    rotation_slider = jp.Input(
        a           = pose3d_figures_view_container,
        id          = "rotation_slider",
        type        = "range",
        min         = "0",
        max         = "359",
        value       = "180",
        step        = "0.01",
        classes     = "w-full"
    )

    def update_pose3d_figures():
        frame_idx = int(frame_slider.value)
        current_data = node_dict["webpage"].current_data
        if current_data is None:
            return
        
        frame               = current_data["datas"][frame_idx]
        motion_world        = frame["pose3d_output"]
        keypoints_scores    = frame["keypoints_scores"]

        motion_world        = np.array(motion_world)
        keypoints_scores    = np.array(keypoints_scores)

        # =================== 3D visualize ===================
        elivation   = float(elivation_slider.value)
        rotation    = float(rotation_slider.value)

        f = plt.figure(figsize=(4, 4))
        ax = f.add_subplot(111, projection='3d')
        pose3d_visualize(ax, motion_world, keypoints_scores, elivation, rotation)

        pose3d_figures.set_figure(f)
        plt.close(f)

        jp.run_task(pose3d_figures.update())

    def update_result():
        frame_idx = int(frame_slider.value)

        current_data = node_dict["webpage"].current_data
        if current_data is None:
            return
        
        frame = current_data["datas"][frame_idx]

        # =================== Image visualize ===================
        img = frame["img_jpeg"]
        img_plane.src = f'data:image/jpeg;base64,{img}'

        update_pose3d_figures()
        jp.run_task(node_dict["webpage"].update())

    def init_slider(frame_count):
        frame_slider.max = str(int(frame_count)-1)
        frame_slider.value = "0"
        frame_slider_info.text = f"Frame {0}/{frame_count}"

        current_data = node_dict["webpage"].current_data
        if current_data is None:
            return
        
        update_result()

    def change_frame(self, msg):
        frame_slider_info.text = f"Frame {frame_slider.value}/{frame_slider.max}"
        update_result()

    def change_pose3d_figures(self, msg):
        elivation_slider_info.text = f"Elevation {elivation_slider.value}/180deg"
        rotation_slider_info.text = f"Rotation {rotation_slider.value}/360deg"
        update_pose3d_figures()

    node_dict["webpage"].init_slider = init_slider
    frame_slider.on("input", change_frame)
    elivation_slider.on("input", change_pose3d_figures)
    rotation_slider.on("input", change_pose3d_figures)

def file_list_view(node_dict: dict):
    file_list_view_container = jp.Div(a=node_dict["plane"], id="file_list_view_container", style="overflow-x: scroll;", classes="w-full h-52 flex items-center border-box border border-green-400 p-5")
    
    def drag_over(self, msg):
        msg.prevent_default = True
    
    file_list_view_container.on("dragover", drag_over)
    def add_file(collected_datas):
        item_container = jp.Div(a=file_list_view_container, style="height: fit-content; width: fit-content; white-space: nowrap;", classes="flex flex-col p-3 cursor-pointer justify-center items-center bg-gray-100 mr-3")
        def item_click(self, msg):
            for component in file_list_view_container.components:
                component.set_class("bg-gray-100")
            self.set_class("bg-green-100")

            try:
                node_dict["webpage"].current_data = None
                for file in node_dict["webpage"].file_list:
                    if file["hash"] == self.hash:
                        node_dict["webpage"].current_data = file
                        break
                
                frame_count = int(collected_datas["frame_count"])
                node_dict["webpage"].init_slider(frame_count)
            except Exception as e:
                print(e)

        item_container.on("click", item_click)

        date = collected_datas["date"]
        frame_count = int(collected_datas["frame_count"])
        fps = collected_datas["fps"]
        batch_size = collected_datas["batch_size"]
        jp.Span(a=item_container, text=date, classes="text-base")
        jp.Span(a=item_container, text=f"Frame Count : {frame_count}", classes="text-base")
        jp.Span(a=item_container, text=f"FPS : {fps}", classes="text-base")
        jp.Span(a=item_container, text=f"Batch Size : {batch_size}", classes="text-base")
        hash = collected_datas["hash"]
        item_container.hash = hash
        
    node_dict["webpage"].add_file = add_file

async def page_ready(self, msg):
    page_id = msg["page_id"]
    websocket_id = msg["websocket_id"]
    session_id = msg["session_id"]
    
    script = "file_list_view_container.addEventListener(\"drop\", async (e) => {e.preventDefault();"
    script += "e[\"page_id\"] = " + str(page_id) + "; "
    script += "e[\"websocket_id\"] = " + str(websocket_id) + "; "
    script += "e[\"session_id\"] = \"" + str(session_id) + "\"; "
    script += "e[\"event_type\"] = \"" + "result_ready" + "\"; " # MUST be result_ready
    script += """
    const chunkSize = 1024 * 512;

    const readFileChunked = (file) => {
        return new Promise((resolve, reject) => {
            const fileReader = new FileReader();
            const chunks = [];
            let offset = 0;

            fileReader.onload = (e) => {
                chunks.push(e.target.result);
                offset += chunkSize;
                const totalChunks = Math.ceil(file.size / chunkSize);
                
                if (offset < file.size) {
                    readNextChunk();
                } else {
                    resolve({ name: file.name, chunks: chunks, totalChunks: totalChunks });
                }
            };
            fileReader.onerror = reject;

            const readNextChunk = () => {
                const blob = file.slice(offset, offset + chunkSize);
                fileReader.readAsText(blob);
            };

            readNextChunk();
        });
    };

    const filePromises = [];
    for (let i = 0; i < e.dataTransfer.files.length; i++) {
        const file = e.dataTransfer.files[i];
        if (file.name.endsWith(".json")) {
            filePromises.push(readFileChunked(file));
        }
    }

    try {
        const files = await Promise.all(filePromises);
        for (const file of files) {
            for (let i = 0; i < file.chunks.length; i++) {
                const chunk = file.chunks[i];
                e.result = { 
                    name: file.name, 
                    chunk: chunk, 
                    totalChunks: file.totalChunks,
                    chunkIndex: i
                }
                await new Promise(resolve => setTimeout(resolve, 1));
                send_to_server(e, "page_event", false);
            }
        }
    } catch (err) {
        console.error("Error processing files:", err);
    }
    });
    """
    jp.run_task(self.run_javascript(script, request_id="get_files"))

async def result_ready(self, msg):
    try:
        result = msg.result
        file_name = result.get("name")
        chunk_data = result.get("chunk")
        total_chunks = result.get("totalChunks")
        chunk_index = result.get("chunkIndex", 0)

        print(total_chunks, chunk_index)
        
        if not all([file_name, chunk_data is not None, total_chunks]):
            return
            
        if file_name not in self.file_chunks:
            self.file_chunks[file_name] = [None] * total_chunks
            
        self.file_chunks[file_name][chunk_index] = chunk_data

        if all(chunk is not None for chunk in self.file_chunks[file_name]):
            data = ''.join(self.file_chunks[file_name])
            
            try:
                parsed_data = json.loads(data)
                self.file_list.append(parsed_data)
                self.add_file(parsed_data)
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Data length: {len(data)}")
                print(f"First 100 chars: {data[:100]}")
            
            del self.file_chunks[file_name]
            
    except Exception as e:
        print(f"Error in result_ready: {e}")

def data_visualization_page():
    wp = jp.WebPage()
    wp.file_chunks = {}
    wp.file_list = []

    wp.node_dict = dict()
    wp.node_dict["webpage"] = wp

    wp.on('page_ready', page_ready)
    wp.on('result_ready', result_ready)

    plane = jp.Div(a=wp, id="plane", classes="h-screen w-screen flex flex-col border-box justify-around p-3")
    wp.node_dict["plane"] = plane
    
    result_view(node_dict=wp.node_dict)
    file_list_view(node_dict=wp.node_dict)

    return wp

if __name__ == "__main__":
    jp.justpy(websockets=False)