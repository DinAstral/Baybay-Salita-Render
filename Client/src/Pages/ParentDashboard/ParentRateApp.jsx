import ParentSidebar from "../../Components/ParentDashboardComponent/ParentSidebar";
import BodyRateApplication from "../../Components/ParentDashboardComponent/BodyRateApplication";

const ParentRateApp = () => {
  return (
    <div className="dashboard">
      <ParentSidebar />
      <div className="dashboard-content">
        <BodyRateApplication />
      </div>
    </div>
  );
};

export default ParentRateApp;
