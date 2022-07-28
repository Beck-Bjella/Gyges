fn _multi_threading_example() {
    let (tx, rx) = mpsc::channel();

    let mut handles = vec![];
    for _ in 0..8 {
        let tx1 = tx.clone();

        let handle = thread::spawn(move || {
            tx1.send(1).unwrap();

        });

        handles.push(handle);

    }

    for h in handles {
        h.join().unwrap();

    }


    let mut scores: Vec<i32> = Vec::new();
    while let Ok(score) = rx.try_recv() {
        scores.push(score);
    }

    println!("{:?}", scores);

}
