import { useState, useEffect } from "react";
import "./Main.css";
import Hero from "../../Components/Home/Hero/Hero";
import Loader from "../../Components/Loader/Loader";
import Title from "../../Components/Home/Title/Title";
import Program from "../../Components/Home/Services/Program";
import About from "../../Components/Home/About/About";
import Contact from "../../Components/Home/Contact/Contact";
import Footer from "../../Components/Home/Footer/Footer";
import Navbar from "../../Components/Home/Nabvar/Navbar";

const Main = () => {
  // Loader state
  const [isLoading, setIsLoading] = useState(true);

  // Function to simulate data fetching or refreshing
  const fetchData = () => {
    console.log("Data refreshed"); // Replace with actual data fetching logic
  };

  useEffect(() => {
    // Initial data loading
    const loadData = () => {
      setTimeout(() => {
        setIsLoading(false);
        fetchData();
      }, 2000); // Initial load delay
    };
    loadData();

    // Immediately refresh data
    setTimeout(() => {
      fetchData();
    }, 2000); // Refresh immediately after initial load
  }, []);

  // Optionally, if you want to log state updates for debugging:
  useEffect(() => {
    console.log("Component re-rendered or data refreshed");
  }, [isLoading]);

  return isLoading ? (
    <Loader />
  ) : (
    <div className="body-main">
      <Navbar />
      <Hero />
      <div className="px-8">
        <Title subTitle="Aming Serbisyo" title="Ang handog namin sa inyo" />
        <Program />
      </div>
      <About />
      <Title
        subTitle="Kami ay kontakin"
        title="Tayu'y maging malapit sa isa't-isa"
      />
      <Contact />
      <Footer />
    </div>
  );
};

export default Main;
