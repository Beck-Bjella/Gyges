use std::fs::File;
use std::io::Write;
use std::fs::OpenOptions;

pub struct TreeStorage {
    nodes: Vec<TreeNode>,
    file: File

}

impl TreeStorage {
    pub fn new(depth: i8) -> TreeStorage {
        let path = format!("src/tree_depth_{}.txt", depth);

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)
            .unwrap();

        return TreeStorage {
            nodes: vec![],
            file

        }

    }

    pub fn end(&mut self) {
        self.save()

    }

    pub fn add(&mut self, node: TreeNode) {
        self.nodes.push(node);

    }

    pub fn add_child_to_node(&mut self, node_id: usize, new_child_id: usize) {
        for node in &mut self.nodes {
            if node.id == node_id {
                node.add_child(new_child_id);

            }

        }

    }

    fn save(&mut self) {
        for node in &self.nodes {
            let mut line = format!("{}|{}|{}|", node.id, node.root, node.leaf);

            for child in &node.children {
                line += child.to_string().as_str();
                line += ","

            }

            writeln!(self.file, "{}", line).unwrap();

        }

        self.file.flush().unwrap();
    
    }

}

#[derive(Debug)]
pub struct TreeNode {
    id: usize,
    children: Vec<usize>,
    // tt_cut: bool,
    // beta_cut: bool,
    // null_mv_cut: bool,
    root: bool,
    leaf: bool

}

impl TreeNode {
    pub fn new(id: usize, root: bool, leaf: bool) -> TreeNode {
        return TreeNode {
            id,
            children: vec![],
            // tt_cut: false,
            // beta_cut: false,
            // null_mv_cut: false,
            root, 
            leaf

        }

    }

    pub fn add_child(&mut self, new_child_id: usize) {
        self.children.push(new_child_id);

    }

}
